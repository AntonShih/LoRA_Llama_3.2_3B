# 🏆 LoRA 微調：鞋子防霉專家 AI

## 📖 專案簡介
本專案使用 **LoRA (Low-Rank Adaptation)** 技術對 **語言模型** 進行微調，專注於 **鞋子防霉知識**，並涵蓋：
- **防霉機制**
- **鞋子材質影響**
- **氣味形成**
- **健康影響**
- **鞋子防滑與性能變化**

📌 **專案目標**
- 提升 AI 在 **鞋子防霉領域的準確性**。
- 優化 AI **回答的流暢度與專業度**。
- **縮短回應時間**，提高模型推理效率。

---

## 📊 LoRA 微調效果比較

| **評估指標** | **LoRA 前** | **LoRA 後** | **提升幅度 (%)** |
|------------|------------|------------|------------|
| **內容完整性** | 5.4 | 7.2 | +33% |
| **關鍵數據準確度** | 4.9 | 6.6 | +35% |
| **語意一致性** | 6.1 | 8.4 | +38% |
| **答案可讀性** | 6.0 | 8.8 | +47% |
| **回應時間 (秒)** | 16.4s | 8.4s | **提升 49%** |

✅ **LoRA 微調後的 AI 在準確性、語意一致性、可讀性、推理速度方面均有顯著提升！**

---

## 🔧 訓練細節與參數設定

```python
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
# from transformers import EarlyStoppingCallback

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
            per_device_train_batch_size = 8,  # 讓模型學更多
            gradient_accumulation_steps = 2,  # 更頻繁更新權重
            num_train_epochs = 13,  # 讓數據學 13 次
            max_steps = -1,  # 讓它依照 epochs 計算步數
            learning_rate = 1.5e-5,  # 降低學習率，讓學習更細緻
            lr_scheduler_type = "cosine_with_restarts",  # 讓學習率下降更平滑
            warmup_steps = 5,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            seed = 8888,
            output_dir = "outputs",
            report_to = "none",
            # save_strategy = "epoch",  # 每個 epoch 儲存 checkpoint
            # evaluation_strategy = "epoch",  # 每個 epoch 評估
            # load_best_model_at_end = True,  # 訓練結束時載入最佳模型
            # metric_for_best_model = "loss",  # 監控 loss 來決定最佳模型
        ),
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] # 如果 3 個 epoch 沒有進步，
    )
