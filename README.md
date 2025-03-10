#  LoRA 微調：鞋子防霉專家 AI Agent

##  專案簡介
本專案使用 **LoRA (Low-Rank Adaptation)** 技術對 **Llama-3.2-3B-Instruct** 進行微調，專注於 **鞋子防霉知識**，並涵蓋：
總共 10 種分類

- **黴菌基礎知識**
- **環境因素**
- **材質影響**
- **存放與保養**
- **使用與清潔**
- **黴菌對鞋子的影響**
- **氣味與感官體驗**
- **健康影響**
- **鞋子特定部位的霉變**
- **性能影響**

 **專案目標**
- 五天內提升 AI 在 **鞋子防霉領域的準確性**。
- 優化 AI **回答的流暢度與專業度**。
- **縮短回應時間**，提高模型推理效率。

---

## 📊LoRA 微調效果比較

| **評估指標** | **LoRA 前** | **LoRA 後** | **提升幅度 (%)** |
|------------|------------|------------|------------|
| **內容完整性** | 5.4 | 7.2 | +33% |
| **關鍵數據準確度** | 4.9 | 6.6 | +35% |
| **語意一致性** | 6.1 | 8.4 | +38% |
| **答案可讀性** | 6.0 | 8.8 | +47% |
| **回應時間 (秒)** | 16.4s | 8.4s | **提升 49%** |

 **LoRA 微調後的 AI 在準確性、語意一致性、可讀性、推理速度方面均有顯著提升！**

---
##  硬體設施挑戰與資料
本專案的開發起點是 **個人電腦 GPU (NVIDIA GTX 1050, 4GB VRAM)**，然而：
- **GTX 1050 無法支援 LoRA 微調**，顯存 **爆滿 (OOM)**，訓練時間估計 **超過 36-48 小時**。
- **FP16/BF16 計算能力不足**，無法有效利用現代 AI 訓練技術。
- **低 VRAM 限制 batch size**，導致每次訓練的運算量極低，訓練速度緩慢。

###  **經過大量研究與實驗，我找到了解決方案**
1️ **轉向 Google Colab (免費T4 GPU)**
   - **解決 VRAM 不足問題**，使 LoRA 訓練可行。
   - **使用高性能雲端 GPU**，讓微調時間縮短 **80%+**。

2️ **使用 [Unsloth](https://github.com/unslothai/unsloth) 進行高效 LoRA 微調**
   - **支援 4-bit 量化**，降低記憶體需求 **40%+**。
   - **比 Hugging Face 官方微調快 2 倍**，大幅節省訓練時間。
   - **內建梯度檢查點 (Gradient Checkpointing)**，適用於 **長序列微調**。

**本專案運用 Google Colab 與 Unsloth 來克服硬體限制，歷經多次測試，才找到高效的 LoRA 微調方法！**

### 資料集設定
- 增加多樣性 : 涵蓋不同類型的黴菌問題和情境
- 階層化組織 : 將資料分為基礎知識、進階診斷、處理方案 3 大類別
    - 在基礎知識加入多樣性設定 : 涵蓋 鞋子黴菌的成因、影響、環境因素、材質差異、人體健康影響
  
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
---
#未來優化建議
.訓練數據優化將100筆資料拓展到1000~10000筆

i增加關鍵數據點
.在訓練數據中**標註**需要 AI 記住的關鍵數據 (如 70-90% 濕度, VOCs 等)。

ii數據擴充
.增加多樣化的表述方式，確保 AI 記住並正確理解關鍵資訊。

iii微調策略改進
.提升學習率
.現有學習率：1.5e-5
.建議改為：2e-5(目前 LoRA 記住了大部分內容，但仍然有部分數據錯誤，適當提升學習率可進一步精準記住細節。)

iv增加微調步數
.現有訓練 Epochs：13
.建議改為：16 (目前 AI 還有部分回答內容缺失，增加 3 個 Epoch 可能讓模型更完整學習數據。)

v啟用驗證集監控、加入 Early Stopping
.注解於參數中，時間有限還未更新數據集，未來有時間更新資料可以嘗試
