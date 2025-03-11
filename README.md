#  LoRA 微調：鞋子防霉專家 AI Agent

##  專案簡介
本專案使用 **LoRA (Low-Rank Adaptation)** 技術對 **Llama-3.2-3B-Instruct** 進行微調，專注於 **鞋子防霉知識**

## **專案目標**
- 五天內提升 LLM 在 **鞋子防霉領域的準確性**。(3/5~3/10)
- 優化 LLM **回答的流暢度與專業度**。
- **縮短回應時間**，提高模型推理效率。

---

##  LoRA 前後量化比較- [必看詳細評測結果](https://blue-caption-57f.notion.site/LoRA-1b2ba6c6f6e180be8ebdcb2578492d0b?pvs=73)

| **評估指標**       | **LoRA 前 平均分數 (1-10)** | **LoRA 後 平均分數 (1-10)** | **提升幅度 (%)** |
|-------------------|--------------------|--------------------|--------------|
| **內容完整性**    | 5.4                | 7.2                | **+33%**     |
| **關鍵數據準確度** | 4.9                | 6.6                | **+35%**     |
| **語意一致性**    | 6.1                | 8.4                | **+38%**     |
| **答案可讀性**    | 6.0                | 8.8                | **+47%**     |
| **回應時間 (秒)** | 16.4 秒            | 8.4 秒             | **提升 49%** |
| **回應精簡程度**  | 5.7                | 8.7                | **+53%**     |

---

###  **關鍵結果：**
- **內容完整性提升 33%** ：LoRA 後模型的回答涵蓋更多核心概念，但仍然部分缺漏，例如 **數據範圍 (如 70-90%)** 與 **特定化學成分 (VOCs, Geosmin)**。

- **關鍵數據準確度提升 35%** ：模型對數據的學習能力增強，但仍可能產生輕微的錯誤資訊 (如錯誤的化學物質)。

- **語意一致性提升 38%** ：微調後模型的邏輯性更佳，避免了不相關的敘述，提高了語言流暢度。

- **答案可讀性提升 47%** ：回答變得更精煉，語句更加清楚，去除了冗餘內容。

- **回應時間縮短 49%** ：LoRA 後的推理效率大幅提升，平均回應時間從 **16.4 秒** 降至 **8.4 秒**，接近 **2 倍加速**。

---
##  硬體設施挑戰
本專案的開發開始是 **個人電腦 GPU (NVIDIA GTX 1050, 4GB VRAM)**，然而：
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
  
---
## 數據集涵蓋 10 大分類
來源:LLM  [數據集已上傳Huggung_face](https://huggingface.co/datasets/AntonShih/FineTome_100k_basic_knowledge_of_mold)

| **編號** | **分類名稱** | **描述** |
|---------|------------|---------|
| **1** | **黴菌基礎知識** | 涵蓋黴菌的生長條件、濕度需求、環境影響等基礎知識。 |
| **2** | **環境因素** | 氣候、濕度、溫度變化如何影響鞋子發霉的可能性。 |
| **3** | **材質影響** | 不同鞋材（皮革、帆布、羊毛等）對黴菌生長的影響。 |
| **4** | **存放與保養** | 如何正確存放與保養鞋子，以防止發霉。 |
| **5** | **使用與清潔** | 運動後、日常穿著後的鞋子清潔方式，避免黴菌滋生。 |
| **6** | **黴菌對鞋子的影響** | 黴菌如何影響鞋子的結構、材質、壽命及外觀。 |
| **7** | **氣味與感官體驗** | 鞋子發霉產生的異味來源，以及如何去除異味。 |
| **8** | **健康影響** | 黴菌可能對人體健康造成的影響，例如腳氣、皮膚病等。 |
| **9** | **鞋子特定部位的霉變** | 哪些部位（鞋墊、鞋底、鞋內襯等）最容易發霉及其原因。 |
| **10** | **性能影響** | 黴菌對鞋子的防滑性、舒適度、支撐性等性能的影響。 |
---

##  未來優化建議
### ** 1.訓練數據優化**
本次 LoRA 微調僅使用 **100 筆資料**，為了進一步提升 AI 準確度與泛化能力，計畫將訓練數據擴展至 **1000 - 10,000 筆**，並優化數據內容。

#### ** (i) 增加關鍵數據點**
**在訓練數據中標註 AI 需要記住的重要資訊，例如：**
- **最佳防霉濕度範圍：70-90%**
- **鞋子發霉氣味的主要來源 (VOCs, Geosmin)**

#### ** (ii) 數據擴充**
**擴展訓練數據的多樣化表述，確保 AI 記住並理解相同概念的不同表達方式**
- **增加多種說法**，如：
  - 「防霉的最佳濕度是 70-90%」➡ 「濕度超過 90% 時黴菌生長最快，低於 70% 則受抑制」
  - 「鞋底發霉會影響防滑能力」➡ 「鞋底菌膜減少與地面的摩擦力，增加滑倒風險」

---

### ** 2. 微調策略改進**
#### ** (iii) 提升學習率**
- **現有學習率：1.5e-5**
- **建議調整為：2e-5**
**理由**：目前 LoRA 記住了大部分內容，但仍然有部分數據錯誤，適當提升學習率 **更精準記住細節**，減少遺漏資訊。

#### ** (iv) 增加微調步數**
- **現有訓練 Epochs：13**
- **建議調整為：16**
 **理由**：目前 AI **部分回答仍不完整**，增加 3 個 Epoch 可以讓模型更充分學習數據，提升準確度。

---

### ** 3. 啟用驗證集監控、加入 Early Stopping**
 **目前因時間有限，尚未更新數據集，但未來可加入以下優化策略(參數部分已備註在程式碼中)：**
- **啟用驗證集監控 (`evaluation_strategy = "epoch"`)**
- **開啟 Early Stopping (`early_stopping_patience = 3`)**
- **在參數設定中增加 `metric_for_best_model = "loss"`**，確保選擇最優模型。

 **未來當資料集擴充到 1000 - 10,000 筆時，可以重新評估這些策略，進一步提升 LoRA 訓練的穩定性與泛化能力！**

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
