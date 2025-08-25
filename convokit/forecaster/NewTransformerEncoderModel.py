try:
    import torch
    import torch.nn.functional as F
    from datasets import Dataset, DatasetDict
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    TRANSFORMERS_AVAILABLE = True
except (ModuleNotFoundError, ImportError) as e:
    raise ModuleNotFoundError(
        "torch, transformers, or datasets is not currently installed. Run 'pip install convokit[llm]' if you would like to use the NewTransformerEncoderModel."
    ) from e

from torch.utils.data import DataLoader
from transformers import get_scheduler
import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
# from sklearn.metrics import roc_curve
from .forecasterModel import ForecasterModel
from .TransformerForecasterConfig import TransformerForecasterConfig
import shutil


os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_CONFIG = TransformerForecasterConfig(
    output_dir="NewTransformerEncoderModel",
    gradient_accumulation_steps=1,
    per_device_batch_size=4,
    num_train_epochs=1,
    learning_rate=6.7e-6,
    random_seed=1,
    context_mode="normal",
    device="cuda",
)


class NewTransformerEncoderModel(ForecasterModel):
    """
    A ConvoKit Forecaster-adherent implementation of conversational forecasting model based on Transformer Encoder Model (e.g. BERT, RoBERTa, SpanBERT, DeBERTa).
    This class is first used in the paper "Conversations Gone Awry, But Then? Evaluating Conversational Forecasting Models"
    (Tran et al., 2025).

    :param model_name_or_path: The name or local path of the pretrained transformer model to load.
    :param config: (Optional) TransformerForecasterConfig object containing parameters for training and evaluation.
    """

    def __init__(self, model_name_or_path, config=DEFAULT_CONFIG):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            model_max_length=512,
            truncation_side="left",
            padding_side="right",
        )
        self.best_threshold = 0.5
        model_config = AutoConfig.from_pretrained(
            model_name_or_path, num_labels=2, problem_type="single_label_classification"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, ignore_mismatched_sizes=True, config=model_config
        ).to(config.device)
        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)
        self.config = config
        return

    def _context_mode(self, context):
        """
        Select the utterances to include in the input context based on the configured context mode.

        This method determines whether to include the full dialogue context or only
        the current utterance, depending on the value of `self.config.context_mode`.

        Supported modes:
        - "normal": Use the full dialogue context (i.e., all utterances leading up to the current one).
        - "no-context": Use only the current utterance.

        :param context: A context tuple containing `context.context` (prior utterances)
            and `context.current_utterance`.

        :return: A list of utterance objects to be used for tokenization.

        :raises ValueError: If `self.config.context_mode` is not one of the supported values.
        """
        if self.config.context_mode == "normal":
            context_utts = context.context
        elif self.config.context_mode == "no-context":
            context_utts = [context.current_utterance]
        else:
            raise ValueError(
                f"Context mode {self.config.context_mode} is not defined. Valid value must be either 'normal' or 'no-context'."
            )
        return context_utts

    def _tokenize(self, context):
        """
        Tokenize a list of utterances into model-ready input using the class tokenizer.

        This method joins the utterances in the given context using the tokenizer's
        separator token (e.g., `[SEP]`), then tokenizes the resulting. It applies
        padding and truncation to ensure the sequence fits within the model's maximum
        input length.

        :param context: A list of Utterance objects.

        :return: A dictionary containing:
            - 'input_ids': the token IDs for the input sequence
            - 'attention_mask': the attention mask corresponding to the input
        """
        tokenized_context = self.tokenizer.encode_plus(
            text=f" {self.tokenizer.sep_token} ".join([u.text for u in context]),
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        return tokenized_context

    def _context_to_testing_data(self, contexts):
        """
        Convert context tuples into a HuggingFace Dataset formatted for BERT-family models.

        This method processes each context tuple by:
        - Extracting the full conversation history associated with the current utterance
        - Generating a label for the conversation using the provided `self.labeler`
        - Formatting the context according to the model’s context mode
        - Tokenizing the resulting text input
        - Collecting input IDs, attention masks, labels, and utterance IDs

        The result is packaged into a `datasets.Dataset` object suitable for training
        or evaluation with a Transformer-based classification model.

        :param contexts: An iterable of context tuples, each containing a current utterance
            and its conversation history.

        :return: A HuggingFace `Dataset` with fields:
            - 'input_ids': tokenized input sequences
            - 'attention_mask': corresponding attention masks
            - 'labels': ground-truth binary labels
            - 'id': IDs of the current utterances
        """
        pairs = {"id": [], "input_ids": [], "attention_mask": [], "labels": []}
        for context in contexts:
            convo = context.current_utterance.get_conversation()
            label = self.labeler(convo)

            context_utts = self._context_mode(context)
            tokenized_context = self._tokenize(context_utts)
            pairs["input_ids"].append(tokenized_context["input_ids"])
            pairs["attention_mask"].append(tokenized_context["attention_mask"])
            pairs["labels"].append(label)
            pairs["id"].append(context.current_utterance.id)
        return Dataset.from_dict(pairs)

    @torch.inference_mode
    @torch.no_grad
    def _predict(
        self,
        dataset,
        model=None,
        threshold=0.5,
        forecast_prob_attribute_name="forecast_prob",
        forecast_attribute_name="forecast",
    ):
        """
        Generate predictions using the model on the given dataset and return them in a Pandas DataFrame.

        :param dataset: A torch-formatted iterable (e.g., HuggingFace Dataset) where each item contains
            'input_ids', 'attention_mask', and 'id'.
        :param model: (Optional) A PyTorch model for inference. If not provided, `self.model` is used.
        :param threshold: (float) Threshold to convert raw probabilities into binary predictions.
        :param forecast_prob_attribute_name: (Optional) Column name for raw forecast probabilities in the output DataFrame.
            Defaults to "forecast_prob" if not specified.
        :param forecast_attribute_name: (Optional) Column name for binary predictions in the output DataFrame.
            Defaults to "forecast" if not specified.

        :return: A Pandas DataFrame indexed by utterance ID. Contains two columns:
            - One with raw probabilities (named `forecast_prob_attribute_name`)
            - One with binary predictions (named `forecast_attribute_name`)
        """
        if not model:
            model = self.model.to(self.config.device)
        utt_ids = []
        preds = []
        scores = []
        for data in tqdm(dataset):
            input_ids = data["input_ids"].to(self.config.device, dtype=torch.long).reshape([1, -1])
            attention_mask = (
                data["attention_mask"].to(self.config.device, dtype=torch.long).reshape([1, -1])
            )
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=-1)
            utt_ids.append(data["id"])
            raw_score = probs[0, 1].item()
            preds.append(int(raw_score > threshold))
            scores.append(raw_score)

        return pd.DataFrame(
            {forecast_attribute_name: preds, forecast_prob_attribute_name: scores}, index=utt_ids
        )

    def _tune_threshold(self, val_dataset, val_contexts):
        """
        Tune the decision threshold and select the best model checkpoint based on validation accuracy.

        This method evaluates all model checkpoints in the configured output directory using a
        held-out validation set.

        The selected model, threshold, and associated metadata are stored in:
        - `self.model`: the best-performing fine-tuned model
        - `self.best_threshold`: the optimal decision threshold
        - `dev_config.json`: file containing best checkpoint metadata
        - `val_predictions.csv`: CSV file with forecast outputs on the validation set

        Additionally, all non-optimal model checkpoints are removed to save disk space, and the
        tokenizer is saved to the directory of the best checkpoint.

        :param val_dataset: A HuggingFace-compatible dataset containing features for validation.
        :param val_contexts: An iterable of context tuples corresponding to the validation set.
                            Used to map utterance IDs to conversation IDs and extract ground-truth labels.

        :return: A dictionary containing the best checkpoint path, best threshold, and best validation accuracy.
        """
        checkpoints = [cp for cp in os.listdir(self.config.output_dir) if "checkpoint-" in cp]
        best_val_accuracy = 0
        val_convo_ids = set()
        utt2convo = {}
        val_labels_dict = {}
        for context in val_contexts:
            convo_id = context.conversation_id
            utt_id = context.current_utterance.id
            label = self.labeler(context.current_utterance.get_conversation())
            utt2convo[utt_id] = convo_id
            val_labels_dict[convo_id] = label
            val_convo_ids.add(convo_id)
        val_convo_ids = list(val_convo_ids)
        for cp in checkpoints:
            full_model_path = os.path.join(self.config.output_dir, cp)
            finetuned_model = AutoModelForSequenceClassification.from_pretrained(
                full_model_path
            ).to(self.config.device)
            val_scores = self._predict(val_dataset, model=finetuned_model)
            # for each CONVERSATION, whether or not it triggers will be effectively determined by what the highest score it ever got was
            highest_convo_scores = {convo_id: -1 for convo_id in val_convo_ids}
            for utt_id in val_scores.index:
                convo_id = utt2convo[utt_id]
                utt_score = val_scores.loc[utt_id].forecast_prob
                if utt_score > highest_convo_scores[convo_id]:
                    highest_convo_scores[convo_id] = utt_score

            val_labels = np.asarray([int(val_labels_dict[c]) for c in val_convo_ids])
            val_scores = np.asarray([highest_convo_scores[c] for c in val_convo_ids])
            # use scikit learn to find candidate threshold cutoffs
            # _, _, thresholds = roc_curve(val_labels, val_scores)
            thresholds = [0.5]

            def acc_with_threshold(y_true, y_score, thresh):
                y_pred = (y_score > thresh).astype(int)
                return (y_pred == y_true).mean()

            accs = [acc_with_threshold(val_labels, val_scores, t) for t in thresholds]
            best_acc_idx = np.argmax(accs)

            print("Accuracy:", cp, accs[best_acc_idx])
            if accs[best_acc_idx] > best_val_accuracy:
                best_checkpoint = cp
                best_val_accuracy = accs[best_acc_idx]
                self.best_threshold = thresholds[best_acc_idx]
                self.model = finetuned_model

        eval_forecasts_df = self._predict(val_dataset, threshold=self.best_threshold)
        eval_prediction_file = os.path.join(self.config.output_dir, "val_predictions.csv")
        eval_forecasts_df.to_csv(eval_prediction_file)

        # Save the best config
        best_config = {}
        best_config["best_checkpoint"] = best_checkpoint
        best_config["best_threshold"] = self.best_threshold
        best_config["best_val_accuracy"] = best_val_accuracy
        config_file = os.path.join(self.config.output_dir, "dev_config.json")
        with open(config_file, "w") as outfile:
            json_object = json.dumps(best_config, indent=4)
            outfile.write(json_object)

        # Clean other checkpoints to save disk space.
        for root, _, _ in os.walk(self.config.output_dir):
            if ("checkpoint" in root) and (best_checkpoint not in root):
                print("Deleting:", root)
                shutil.rmtree(root)
        # Save the tokenizer.
        self.tokenizer.save_pretrained(
            os.path.join(self.config.output_dir, best_config["best_checkpoint"])
        )
        return best_config

    def _context_to_training_data(self, contexts, k):
        """
        Convert context tuples into a HuggingFace Dataset formatted for training with multiple turns.

        This method processes each context tuple by:
        - Extracting the last k timestamps (with full conversational history/context upto that timestamp) of the conversation
        - Generating a label for the conversation using the provided `self.labeler`
        - Tokenizing the resulting k contexts associated with the k timestamps
        - Padding sequences in two dimensions: sequence length and number of timestamps (k).
            Some conversations may have fewer than k timestamps, in which case we pad with copies of the earliest timestamp.

        :param contexts: An iterable of context tuples, each containing a current utterance
            and its conversation history.
        :param k: The number of turns to include in the input.

        :return: A HuggingFace `Dataset` with fields:
            - 'input_ids': tokenized input sequences for k turns
            - 'attention_mask': corresponding attention masks
            - 'labels': ground-truth binary labels
            - 'id': IDs of the current utterances
        """
        pairs = {"id": [], "input_ids": [], "attention_mask": [], "labels": []}
        for context in contexts:
            convo = context.current_utterance.get_conversation()
            label = self.labeler(convo)

            # Generate k timestamps/contexts
            context_utts = [context.context[:max(1, len(context.context) + i)] for i in range(-k+1, 1)]
            tokenized_contexts = [self._tokenize(utt) for utt in context_utts]

            # Pad sequences
            input_ids = np.array([tc["input_ids"] for tc in tokenized_contexts])
            attention_mask = np.array([tc["attention_mask"] for tc in tokenized_contexts])

            pairs["input_ids"].append(input_ids)
            pairs["attention_mask"].append(attention_mask)
            pairs["labels"].append(label)
            pairs["id"].append(context.current_utterance.id)
        return Dataset.from_dict(pairs)

    def fit(self, contexts, val_contexts):
        """
        Fine-tune the TransformerEncoder model with DataLoader batches (FP32 only).
        Best checkpoint is selected by _tune_threshold() on the held-out validation set.
        """
        # ----------------------------
        # Defaults (best-practice fallbacks if missing in config)
        # ----------------------------
        cfg = self.config
        k = getattr(cfg, "conversation_training_length", 8)
        accum_steps = getattr(cfg, "gradient_accumulation_steps", 4)
        per_device_batch_size = getattr(cfg, "per_device_batch_size", 1)
        weight_decay = getattr(cfg, "weight_decay", 0.01)
        lr_scheduler_type = getattr(cfg, "lr_scheduler_type", "linear")
        warmup_ratio = getattr(cfg, "warmup_ratio", 0.06)
        max_grad_norm = getattr(cfg, "max_grad_norm", 1.0)
        num_workers = getattr(cfg, "dataloader_num_workers", min(4, max(1, (os.cpu_count() or 2) // 2)))
        pin_memory = True if torch.cuda.is_available() else False

        # ----------------------------
        # Build datasets and loaders
        # ----------------------------
        val_contexts = list(val_contexts)
        train_ds = self._context_to_training_data(contexts, k)
        val_ds = self._context_to_testing_data(val_contexts)
        dataset = DatasetDict({"train": train_ds, "val_for_tuning": val_ds})
        dataset.set_format("torch")

        def collate_fn(examples):
            return {
                "input_ids": torch.stack([torch.as_tensor(ex["input_ids"]) for ex in examples]),
                "attention_mask": torch.stack([torch.as_tensor(ex["attention_mask"]) for ex in examples]),
                "labels": torch.as_tensor([ex["labels"] for ex in examples]),
                "id": [ex["id"] for ex in examples],
            }

        train_loader = DataLoader(
            dataset["train"],
            batch_size=per_device_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

        # ----------------------------
        # Optimizer (wd hygiene) + Scheduler
        # ----------------------------
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "ln_f.weight", "embeddings.word_embeddings.weight"]
        param_groups = [
            {
                "params": [p for n, p in self.model.named_parameters()
                        if p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                        if p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=cfg.learning_rate, betas=(0.9, 0.999), eps=1e-8)

        steps_per_epoch = max(1, len(train_loader) // max(1, accum_steps) + (1 if (len(train_loader) % max(1, accum_steps)) else 0))
        total_steps = steps_per_epoch * cfg.num_train_epochs
        warmup_steps = int(warmup_ratio * total_steps)

        scheduler = get_scheduler(
            name=lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # ----------------------------
        # Training loop (FP32 only)
        # ----------------------------
        self.model.train()
        global_step = 0
        pbar = tqdm(total=total_steps, desc="Training", leave=True)

        for epoch in range(cfg.num_train_epochs):
            running_loss = 0.0
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(cfg.device)           # (B, k, L)
                attention_mask = batch["attention_mask"].to(cfg.device) # (B, k, L)
                labels = batch["labels"].to(cfg.device)                 # (B,)

                B, K, L = input_ids.shape
                input_ids_flat = input_ids.view(B * K, L)
                attention_mask_flat = attention_mask.view(B * K, L)

                outputs = self.model(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
                logits = outputs.logits.view(B, K, -1)  # (B, k, 2)

                # compute per-example loss and average
                per_example_losses = []
                for b in range(B):
                    per_example_losses.append(self.compute_loss(logits[b], labels[b], epoch))
                batch_loss = torch.stack(per_example_losses).mean() / max(1, accum_steps)

                batch_loss.backward()
                running_loss += batch_loss.item() * max(1, accum_steps)

                # Step every accumulation boundary
                is_update_step = ((step + 1) % max(1, accum_steps) == 0) or ((step + 1) == len(train_loader))
                if is_update_step:
                    if max_grad_norm is not None and max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                    global_step += 1
                    pbar.set_postfix(step=global_step, epoch=epoch + 1, lr=scheduler.get_last_lr()[0])
                    pbar.update(1)

            self.save_model(global_step)
            avg_loss = running_loss / max(1, len(train_loader.dataset))
            print(f"Epoch {epoch+1}/{cfg.num_train_epochs} - train_loss: {avg_loss:.4f}")

        pbar.close()
        _ = self._tune_threshold(dataset["val_for_tuning"], val_contexts)
        return


    def save_model(self, global_step):
        """
        Save the fine-tuned model and tokenizer to the specified output directory.

        :param output_dir: (Optional) Directory where the model and tokenizer will be saved.
            If not provided, defaults to `self.config.output_dir`.
        """
        output_dir = self.config.output_dir
        step_dir = f"{output_dir}/checkpoint-{global_step}"
        self.model.save_pretrained(step_dir, safe_serialization=True)     # model.safetensors + config.json
        self.tokenizer.save_pretrained(step_dir)                          # tokenizer files
        return

    def compute_loss(self, logits, labels, epoch):
        """
        New training loss over k utterances:
        - Supervise the *last* turn with the gold label (standard CE).
        - For turns 0..k-2, add a consistency loss that makes logits[t]
            match the soft targets from logits[t+1] (teacher).
        - Additionally, add a loss that makes the turn with the highest
            positive logit match the gold label.

        Args:
            logits: Tensor of shape (k, 2) with raw (pre-softmax) scores.
            labels: Scalar int in {0,1} or a tensor (ignored except for last-turn CE if length>1).

        Returns:
            Scalar tensor combining last-turn CE and consistency loss.
        """

        k = logits.size(0)

        # ----- 1) Last-turn supervised loss (standard CE vs gold) -----
        last_logits = logits[-1]  # (2,)
        loss_last = F.cross_entropy(last_logits.unsqueeze(0), labels.view(1))

        # ----- 2) Consistency loss (use logits[t+1] as soft labels for logits[t]) -----
        # if k >= 2:
        #     student_logits = logits[:-1]        # (k-1, 2)
        #     teacher_labels = torch.argmax(logits[1:], dim=-1)  # (k-1,)
        #     consistency_loss = F.cross_entropy(student_logits, teacher_labels)
        # else:
        #     consistency_loss = torch.zeros((), device=logits.device)
        # ----- 3) Highest Logit Loss  -----
        highest_logits = logits[torch.argmax(F.softmax(logits, dim=-1)[:, 1])] # (2,)
        loss_highest = F.cross_entropy(highest_logits.unsqueeze(0), labels.unsqueeze(0))

        return 1/(1+epoch) * (loss_last + loss_highest*epoch)  # Scheduling to reduce the focus on last-turn CE over time

    def transform(self, contexts, forecast_attribute_name, forecast_prob_attribute_name):
        """
        Generate forecasts using the fine-tuned TransformerEncoder model on the provided contexts, and save the predictions to the output directory specified in the configuration.

        :param contexts: context tuples from the Forecaster framework
        :param forecast_attribute_name: Forecaster will use this to look up the table column containing your model's discretized predictions (see output specification below)
        :param forecast_prob_attribute_name: Forecaster will use this to look up the table column containing your model's raw forecast probabilities (see output specification below)

        :return: a Pandas DataFrame, with one row for each context, indexed by the ID of that context's current utterance. Contains two columns, one with raw probabilities named according to forecast_prob_attribute_name, and one with discretized (binary) forecasts named according to forecast_attribute_name
        """
        test_pairs = self._context_to_testing_data(contexts)
        dataset = DatasetDict({"test": test_pairs})
        dataset.set_format("torch")
        forecasts_df = self._predict(
            dataset["test"],
            threshold=self.best_threshold,
            forecast_attribute_name=forecast_attribute_name,
            forecast_prob_attribute_name=forecast_prob_attribute_name,
        )

        prediction_file = os.path.join(self.config.output_dir, "test_predictions.csv")
        forecasts_df.to_csv(prediction_file)

        return forecasts_df
