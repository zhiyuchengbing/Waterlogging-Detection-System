import os
import sys
import csv
import argparse
import shutil
from pathlib import Path
from typing import Optional

# 固定顶部原图根目录为绝对路径，保持与主程序一致
RUNS_OUTPUT_IMAGES_DIR = r"E:\积水识别项目\demo0625\demo\runs\output_images"

# 复用现有数据库查询函数
try:
    from test_mysql import query_recent_water_carriages
except Exception as e:
    print("错误：无法从 test_mysql 导入 query_recent_water_carriages，请确认文件存在且可用。")
    print(str(e))
    sys.exit(1)


def resolve_original_top_image(train_name: str, carriage_id: str, output_dir: str = RUNS_OUTPUT_IMAGES_DIR) -> Optional[str]:
    """
    根据项目统一规范，原始顶部图片命名为 runs/output_images/<train_name>/frame_<序号>.jpg
    其中 <序号> = int(carriage_id) 以去除可能的前导0
    返回存在的绝对路径，若不存在返回 None
    """
    try:
        num = int(str(carriage_id).strip())
    except ValueError:
        return None

    image_name = f"frame_{num}.jpg"
    path = os.path.join(output_dir, train_name, image_name)
    return os.path.abspath(path) if os.path.exists(path) else None


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def export_recent_originals(days: int, dest: str, limit: int = 10000, output_dir: str = RUNS_OUTPUT_IMAGES_DIR, overwrite: bool = False) -> dict:
    """
    导出最近 N 天内检测为“有水”的车厢的原始顶部图片到指定目录。

    返回统计信息字典：{"total": int, "resolved": int, "copied": int, "missing": int, "manifest": str}
    """
    records = query_recent_water_carriages(days=days, limit=limit) or []

    ensure_dir(dest)
    manifest_path = os.path.join(dest, "manifest.csv")

    copied = 0
    resolved = 0
    missing = 0

    # 打开清单文件
    with open(manifest_path, "w", newline="", encoding="utf-8-sig") as mf:
        writer = csv.writer(mf)
        writer.writerow([
            "train_name",
            "carriage_id",
            "full_carriage_number",
            "recognition_time",
            "water_area_ratio",
            "original_image_path",
            "copied_to"
        ])

        for rec in records:
            train_name = str(rec.get("train_name", "")).strip()
            carriage_id = str(rec.get("carriage_id", "")).strip()
            full_carriage_number = str(rec.get("full_carriage_number", "")).strip()
            recog_time = str(rec.get("recognition_time", "")).strip()
            water_area_ratio = rec.get("water_area_ratio", 0)

            original_path = resolve_original_top_image(train_name, carriage_id, output_dir=output_dir)
            copied_to = ""
            if original_path:
                resolved += 1
                # 目标目录结构：<dest>/<train_name>/frame_<n>.jpg
                target_dir = os.path.join(dest, train_name)
                ensure_dir(target_dir)
                target_path = os.path.join(target_dir, os.path.basename(original_path))

                try:
                    if not os.path.exists(target_path) or overwrite:
                        shutil.copy2(original_path, target_path)
                    copied += 1
                    copied_to = os.path.abspath(target_path)
                except Exception as e:
                    # 拷贝失败当作缺失记录统计
                    missing += 1
                    copied_to = f"COPY_FAILED: {e}"
            else:
                missing += 1

            writer.writerow([
                train_name,
                carriage_id,
                full_carriage_number,
                recog_time,
                water_area_ratio,
                original_path or "NOT_FOUND",
                copied_to
            ])

    return {
        "total": len(records),
        "resolved": resolved,
        "copied": copied,
        "missing": missing,
        "manifest": os.path.abspath(manifest_path),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="导出最近 N 天内有积水车厢的原始顶部图片（未处理前）到指定目录"
    )
    parser.add_argument("--days", type=int, default=100, help="查询最近N天（默认30）")
    parser.add_argument("--dest", default="F:\\积水数据1", help="导出目标目录(默认 F:\\积水数据)")
    parser.add_argument("--limit", type=int, default=10000, help="最大记录数（默认10000）")
    parser.add_argument("--output-dir", default=RUNS_OUTPUT_IMAGES_DIR, help="原始顶部图片根目录（默认 runs/output_images）")
    parser.add_argument("--overwrite", action="store_true", help="已存在目标文件时是否覆盖")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    stats = export_recent_originals(
        days=args.days,
        dest=args.dest,
        limit=args.limit,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )

    print("==== 导出完成 ====")
    print(f"总记录: {stats['total']}")
    print(f"已解析原图: {stats['resolved']}")
    print(f"已复制: {stats['copied']}")
    print(f"缺失/复制失败: {stats['missing']}")
    print(f"清单: {stats['manifest']}")


if __name__ == "__main__":
    sys.exit(main())
