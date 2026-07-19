"""Ước lượng độ lệch grade từ Accuracy và Quadratic Weighted Kappa.

Lưu ý quan trọng:
    Chỉ Accuracy, QWK và số mẫu không đủ để tính chính xác MAE grade.
    Khi không có phân bố nhãn thật/dự đoán, chương trình giả định các lớp
    cân bằng. Muốn có kết quả chính xác, hãy tính trực tiếp từ y_true/y_pred.

Ví dụ chạy với kết quả mặc định của mô hình:
    python estimate_grade_error.py

Truyền kết quả khác:
    python estimate_grade_error.py --accuracy 0.85918 --qwk 0.92520 --samples 1000
"""

from __future__ import annotations

import argparse
import math
import sys


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def balanced_expected_disagreement(num_classes: int) -> float:
    """Tính D_e khi nhãn thật và nhãn dự đoán phân bố đều."""
    max_distance = num_classes - 1
    total = 0.0

    for true_grade in range(num_classes):
        for predicted_grade in range(num_classes):
            total += ((true_grade - predicted_grade) / max_distance) ** 2

    return total / (num_classes * num_classes)


def mae_bounds_from_mse(
    mean_squared_distance: float,
    max_distance: int,
) -> tuple[float, float]:
    """Tìm khoảng MAE có thể có từ MSE với độ lệch nguyên 1..max_distance.

    QWK cho biết thông tin về bình phương khoảng cách. Nhiều phân bố lỗi khác
    nhau có thể có cùng MSE nhưng MAE khác nhau, vì vậy chỉ suy ra được một
    khoảng giá trị chứ không có một MAE duy nhất.
    """
    if not 1.0 <= mean_squared_distance <= max_distance**2:
        raise ValueError(
            "MSE trên ca sai nằm ngoài miền hợp lệ. "
            "Giả định phân bố cân bằng có thể không phù hợp."
        )

    candidates: list[float] = []

    for lower in range(1, max_distance + 1):
        for upper in range(lower, max_distance + 1):
            lower_sq = lower**2
            upper_sq = upper**2

            if lower == upper:
                if math.isclose(mean_squared_distance, lower_sq):
                    candidates.append(float(lower))
                continue

            if lower_sq <= mean_squared_distance <= upper_sq:
                upper_ratio = (
                    (mean_squared_distance - lower_sq) / (upper_sq - lower_sq)
                )
                mean_absolute_distance = (
                    (1.0 - upper_ratio) * lower + upper_ratio * upper
                )
                candidates.append(mean_absolute_distance)

    if not candidates:
        raise ValueError("Không thể suy ra khoảng MAE từ các tham số đã cho.")

    return min(candidates), max(candidates)


def estimate_grade_error(
    accuracy: float,
    qwk: float,
    num_samples: int,
    num_classes: int = 5,
    expected_disagreement: float | None = None,
) -> dict[str, float]:
    """Ước lượng độ lệch grade theo giả định đã chọn."""
    if not 0.0 <= accuracy < 1.0:
        raise ValueError("Accuracy phải nằm trong [0, 1).")
    if not -1.0 <= qwk <= 1.0:
        raise ValueError("QWK phải nằm trong [-1, 1].")
    if num_samples <= 0:
        raise ValueError("Số mẫu phải lớn hơn 0.")
    if num_classes < 2:
        raise ValueError("Số lớp phải từ 2 trở lên.")

    if expected_disagreement is None:
        expected_disagreement = balanced_expected_disagreement(num_classes)

    error_rate = 1.0 - accuracy
    estimated_correct = num_samples * accuracy
    estimated_wrong = num_samples * error_rate

    # QWK = 1 - D_o / D_e  =>  D_o = (1 - QWK) * D_e
    observed_weighted_disagreement = (1.0 - qwk) * expected_disagreement

    # Trọng số QWK đã chuẩn hóa cho (K - 1)^2, nên nhân ngược lại để
    # thu được tổng bình phương khoảng cách grade thực tế.
    max_distance = num_classes - 1
    squared_error_sum = (
        num_samples
        * max_distance**2
        * observed_weighted_disagreement
    )
    mse_wrong = squared_error_sum / estimated_wrong
    rmse_wrong = math.sqrt(mse_wrong)
    mae_wrong_min, mae_wrong_max = mae_bounds_from_mse(
        mse_wrong,
        max_distance,
    )

    return {
        "expected_disagreement": expected_disagreement,
        "estimated_correct": estimated_correct,
        "estimated_wrong": estimated_wrong,
        "squared_error_sum": squared_error_sum,
        "mse_wrong": mse_wrong,
        "rmse_wrong": rmse_wrong,
        "mae_wrong_min": mae_wrong_min,
        "mae_wrong_max": mae_wrong_max,
        "mae_wrong_midpoint": (mae_wrong_min + mae_wrong_max) / 2.0,
        "mae_all_min": error_rate * mae_wrong_min,
        "mae_all_max": error_rate * mae_wrong_max,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ước lượng độ lệch grade từ Accuracy và QWK."
    )
    parser.add_argument(
        "--accuracy",
        type=float,
        default=0.8578575632725133,
        help="Accuracy dạng thập phân, ví dụ 0.8578.",
    )
    parser.add_argument(
        "--qwk",
        type=float,
        default=0.9262127204735553,
        help="Quadratic Weighted Kappa.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Số lượng mẫu test/validation.",
    )
    parser.add_argument(
        "--classes",
        type=int,
        default=5,
        help="Số grade, mặc định 5 cho DR grade 0..4.",
    )
    parser.add_argument(
        "--expected-disagreement",
        type=float,
        default=None,
        help=(
            "D_e thực tế tính từ phân bố nhãn thật/dự đoán. "
            "Nếu bỏ trống, dùng giả định các lớp cân bằng."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = estimate_grade_error(
        accuracy=args.accuracy,
        qwk=args.qwk,
        num_samples=args.samples,
        num_classes=args.classes,
        expected_disagreement=args.expected_disagreement,
    )

    assumption = (
        "phân bố lớp cân bằng"
        if args.expected_disagreement is None
        else "D_e do người dùng cung cấp"
    )

    print("=== ƯỚC LƯỢNG ĐỘ LỆCH GRADE ===")
    print(f"Accuracy:                  {args.accuracy:.6f}")
    print(f"QWK:                       {args.qwk:.6f}")
    print(f"Số mẫu:                    {args.samples}")
    print(f"Giả định:                   {assumption}")
    print(f"D_e:                        {result['expected_disagreement']:.6f}")
    print()
    print(
        "Số ca đúng ước tính:       "
        f"{result['estimated_correct']:.2f} (~{round(result['estimated_correct'])})"
    )
    print(
        "Số ca sai ước tính:        "
        f"{result['estimated_wrong']:.2f} (~{round(result['estimated_wrong'])})"
    )
    print(f"Tổng bình phương sai lệch:  {result['squared_error_sum']:.4f}")
    print(f"MSE trên riêng ca sai:      {result['mse_wrong']:.4f}")
    print(f"RMSE trên riêng ca sai:     {result['rmse_wrong']:.4f} grade")
    print(
        "Khoảng MAE trên ca sai:    "
        f"{result['mae_wrong_min']:.4f}–{result['mae_wrong_max']:.4f} grade"
    )
    print(
        "Ước lượng giữa khoảng:     "
        f"{result['mae_wrong_midpoint']:.4f} grade"
    )
    print(
        "Khoảng MAE trên toàn tập:  "
        f"{result['mae_all_min']:.4f}–{result['mae_all_max']:.4f} grade"
    )
    print()
    print(
        "Lưu ý: MAE chính xác chỉ tính được khi có y_true và y_pred; "
        "Accuracy + QWK + số mẫu không xác định duy nhất phân bố khoảng cách lỗi."
    )


if __name__ == "__main__":
    main()
