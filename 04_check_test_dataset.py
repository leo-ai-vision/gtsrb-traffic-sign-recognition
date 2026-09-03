from pathlib import Path

from src.dataset import collect_gtsrb_test_samples


PROJECT_DIR = Path(__file__).resolve().parent

TEST_IMAGE_DIR = (
    PROJECT_DIR
    / "data"
    / "GTSRB_Final_Test_Images"
    / "GTSRB"
    / "Final_Test"
    / "Images"
)

TEST_CSV_PATH = (
    PROJECT_DIR
    / "data"
    / "GTSRB_Final_Test_GT"
    / "GT-final_test.csv"
)


def main():
    # 读取测试图片路径和对应标签
    test_image_paths, test_labels = (
        collect_gtsrb_test_samples(
            test_image_dir=TEST_IMAGE_DIR,
            test_csv_path=TEST_CSV_PATH
        )
    )

    # 检查CSV中的图片是否真实存在
    missing_count = 0

    for image_path in test_image_paths:
        if not image_path.exists():
            missing_count += 1

    print("测试图片数量：", len(test_image_paths))
    print("测试标签数量：", len(test_labels))
    print("测试集类别数量：", len(set(test_labels)))
    print("不存在的图片数量：", missing_count)

    print("第一张测试图片：", test_image_paths[0])
    print("第一张测试图片标签：", test_labels[0])

    print("最后一张测试图片：", test_image_paths[-1])
    print("最后一张测试图片标签：", test_labels[-1])


if __name__ == "__main__":
    main()