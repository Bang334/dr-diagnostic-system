# DR Ensemble Report Files

Tai lieu nay tong hop cac file nen giu cho phan bao cao va demo ensemble
EfficientNetB3 + ConvNeXt Tiny.

## Pipeline chinh nen dua vao bao cao

| File | Vai tro |
| --- | --- |
| `ai/grading/dataset_CLAHE.py` | One-cell Colab xu ly anh offline: RGB crop, Ben Graham light, CLAHE LAB. Day la file giai thich buoc tien xu ly anh voong mac. |
| `ai/grading/make_clean_split.py` | Tao split sach theo image id de giam leakage giua train/val/test. |
| `ai/grading/colab_train_b3_clahe_one_cell.py` | Train/fine-tune EfficientNetB3 tren dataset CLAHE LAB, co weighted CE, ordinal EMD, progressive resize va xuat bao cao danh gia. |
| `ai/grading/colab_train_convnext_rgb_crop_one_cell.py` | Train/fine-tune ConvNeXt Tiny tren dataset RGB crop, dung checkpoint tot nhat de tiep tuc cai thien. |
| `ai/grading/colab_train_b4_rgb_crop_one_cell.py` | Baseline EfficientNetB4 tren RGB crop. Nen neu trong bao cao nhu baseline so sanh, khong phai model ensemble chinh. |
| `ai/ensemble_test/colab_test_b3_convnext_ensemble_one_cell.py` | Test ensemble B3 + ConvNeXt tren Colab, chon weight tren validation va danh gia test. |
| `ai/ensemble_test/local_api.py` | API Swagger local de upload anh fundus va xem ket qua du doan cua B3, ConvNeXt va ensemble. |
| `ai/ensemble_test/run_local.ps1` | Lenh khoi dong Swagger API tren Windows. |
| `ai/ensemble_test/requirements-local.txt` | Dependency toi thieu cho Swagger API local. |

## Dataset va checkpoint can ghi trong bao cao

| Thanh phan | Duong dan/thong tin |
| --- | --- |
| Dataset train noi bo | Dataset fundus da xu ly offline: `rgb_crop_512_clean_split` va `clahe_lab_512_clean_split`. |
| Dataset test doc lap | DDR/OIA-DDR da phan lop: `F:\Downloads\DDR-dataset-parts\DDR-dataset\DDR_DR_grading_classified`. |
| Lop bi loai khi test 5 grade | `ungradable` cua DDR, tuong ung label 5, khong thuoc thang DR 0-4. |
| B3 checkpoint | `F:\Downloads\b3\checkpoint-best.pth` hoac checkpoint B3 tot nhat tren Google Drive. |
| ConvNeXt checkpoint | `F:\Downloads\convnet\checkpoint-best.pth` hoac checkpoint ConvNeXt tot nhat tren Google Drive. |
| Ensemble weight da test | B3 weight `0.4`, ConvNeXt weight `0.6` neu dung ket qua validation ensemble gan nhat. |

## File nen giu nhung khong can tap trung trong bao cao

| File | Ly do |
| --- | --- |
| `ai/grading/train.py` | Script training RETFound/duong train tong quat co san cua project. Giu de doi chieu, khong phai pipeline ensemble B3 + ConvNeXt chinh. |
| `ai/grading/evaluate_test.py` | Script danh gia checkpoint rieng le. Co ich neu can test tung model, nhung ensemble da co script rieng. |
| `ai/grading/main.py` | API grading goc cua he thong. Giu neu backend hien tai dang dung. |
| `ai/grading/model_handler.py` | Handler inference goc cua he thong. Giu neu backend hien tai dang dung. |
| `ai/grading/README.md` | Mo ta training ban dau cua module grading. |

## File/thu muc du thua nen khong dua vao bao cao

| File/thu muc | Xu ly de xuat |
| --- | --- |
| `ai/grading/tmp.py` | File nhap/thu nghiem cu, da nam trong `.gitignore`; co the xoa local. |
| `ai/ensemble_test/.venv/` | Moi truong Python local, khong commit va co the tao lai khi chay `run_local.ps1`. |
| `ai/ensemble_test/__pycache__/` | Cache Python, co the xoa bat cu luc nao. |
| `ai/grading/DR_Training_Colab.ipynb` | Notebook cu; chi giu neu thay/co van yeu cau minh hoa lich su thu nghiem. |
| `ai/grading/DR_Test_Only_Colab.ipynb` | Notebook cu; khong can neu da dung script one-cell va ensemble test moi. |

## Lenh chay Swagger local

Neu dang o repo root:

```powershell
cd F:\STUDY\TTTN\dr-diagnostic-system
powershell -ExecutionPolicy Bypass -File .\ai\ensemble_test\run_local.ps1
```

Neu dang o thu muc `ai`:

```powershell
powershell -ExecutionPolicy Bypass -File .\ensemble_test\run_local.ps1
```

Mo Swagger tai:

```text
http://127.0.0.1:8010/docs
```

## Lenh test DDR bang Swagger

Upload anh trong cac thu muc:

```text
F:\Downloads\DDR-dataset-parts\DDR-dataset\DDR_DR_grading_classified\test\0
F:\Downloads\DDR-dataset-parts\DDR-dataset\DDR_DR_grading_classified\test\1
F:\Downloads\DDR-dataset-parts\DDR-dataset\DDR_DR_grading_classified\test\2
F:\Downloads\DDR-dataset-parts\DDR-dataset\DDR_DR_grading_classified\test\3
F:\Downloads\DDR-dataset-parts\DDR-dataset\DDR_DR_grading_classified\test\4
```

Khong dung thu muc `ungradable` khi tinh accuracy 5 lop.
