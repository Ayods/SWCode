import os
import shutil
import cv2
import numpy as np
from deepface import DeepFace

def loadAndFilter(data_dir, min_images=2, max_people=500):
    qualified = []
    count = 0
    for person in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person)
        if os.path.isdir(person_dir):
            images = os.listdir(person_dir)
            if len(images) >= min_images:
                count += 1
                if count > max_people:
                    break
                # 只记录每个合格人物的前两张图片
                qualified.extend([{"path": os.path.join(person_dir, img_file), "name": person} for img_file in images[:2]])
    print(f"Selected {count} qualified individuals with {len(qualified)} total images.")
    return qualified

def classifyByGender(image_paths):
    male_images = []
    female_images = []
    classified_names = set()

    for image in image_paths:
        if image["name"] in classified_names: 
            continue
        try:
            analysis = DeepFace.analyze(img_path=image["path"], actions=["gender"], enforce_detection=False)
            if isinstance(analysis, list):
                analysis = analysis[0]
            if isinstance(analysis, dict) and "dominant_gender" in analysis:
                gender = analysis["dominant_gender"]
                classified_names.add(image["name"])
                if gender == "Man":
                    male_images.extend([img for img in image_paths if img["name"] == image["name"]])
                elif gender == "Woman":
                    female_images.extend([img for img in image_paths if img["name"] == image["name"]])
            else:
                print(f"Unexpected result format for {image['path']}: {analysis}")

        except Exception as e:
            print(f"Error processing {image['path']}: {str(e)}")


    return male_images, female_images

def classifyByAge(image_paths):
    age_groups = {"0-18": [], "19-35": [], "36-50": [], "51+": []}
    classified_names = set()
    for image in image_paths:
        if image["name"] in classified_names:
            continue
        try:
            analysis = DeepFace.analyze(img_path=image["path"], actions=["age"], enforce_detection=False)
            if isinstance(analysis, list):
                analysis = analysis[0]
            if isinstance(analysis, dict) and "age" in analysis:
                age = analysis["age"]
                age_group = categorizeAge(age)
                classified_names.add(image["name"])
                age_groups[age_group].extend([img for img in image_paths if img["name"] == image["name"]])
            else:
                print(f"Unexpected result format for {image['path']}: {analysis}")

        except Exception as e:
            print(f"Error processing {image['path']}: {str(e)}")
    return age_groups

def categorizeAge(age):
    if age <= 18:
        return "0-18"
    elif 19 <= age <= 35:
        return "19-35"
    elif 36 <= age <= 50:
        return "36-50"
    else:
        return "51+"

def classifyByNationality(image_paths):
    nationality_groups = {}
    classified_names = set()

    for image in image_paths:
        if image["name"] in classified_names:
            continue
        try:
            analysis = DeepFace.analyze(img_path=image["path"], actions=["race"], enforce_detection=False)
            if isinstance(analysis, list):
                analysis = analysis[0]
            if isinstance(analysis, dict) and "dominant_race" in analysis:
                race = analysis["dominant_race"]
                nationality = raceToNationality(race)
                classified_names.add(image["name"])
                if nationality not in nationality_groups:
                    nationality_groups[nationality] = []
                nationality_groups[nationality].extend([img for img in image_paths if img["name"] == image["name"]])
            else:
                print(f"Unexpected result format for {image['path']}: {analysis}")

        except Exception as e:
            print(f"Error processing {image['path']}: {str(e)}")


    return nationality_groups

def raceToNationality(race):
    mapping = {
        "asian": "Asian",
        "white": "European",
        "black": "African",
        "latino hispanic": "South American",
        "middle eastern": "Middle Eastern",
        "indian": "Indian"
    }
    return mapping.get(race.lower(), "Unknown")

def augmentImages(image_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    dark_images = []
    blur_images = []
    processed = set()
    for img1, img2 in zip(image_paths[::2], image_paths[1::2]):
        if img1["name"] in processed:
            continue
        img_name = os.path.basename(img1["path"])
        dark_path = os.path.join(output_dir, f"dark_{img_name}")
        blur_path = os.path.join(output_dir, f"blur_{img_name}")
        # 生成增强图片
        augmentImage(img1["path"], dark_path, brightness_factor=0.5)
        augmentImage(img1["path"], blur_path, blur=True)

        dark_images.append({"path": dark_path, "name": img1["name"], "original_pair": img2})
        blur_images.append({"path": blur_path, "name": img1["name"], "original_pair": img2})
        processed.add(img1["name"])
    return dark_images, blur_images

def augmentImage(image_path, output_path, brightness_factor=1.0, blur=False):
    img = cv2.imread(image_path)
    if brightness_factor != 1.0:
        img = adjustBrightness(img, brightness_factor)
    if blur:
        img = applyBlur(img)
    cv2.imwrite(output_path, img)

def adjustBrightness(image, factor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def applyBlur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

def test1(image_paths, model_name="VGG-Face"):
    correct = 0
    total = 0
    for image1, image2 in zip(image_paths[::2], image_paths[1::2]):
        try:
            result = DeepFace.verify(img1_path=image1["path"], img2_path=image2["path"], model_name=model_name, enforce_detection=False)
            if result["verified"]:
                correct += 1
            total += 1
        except Exception as e:
            print(f"Error testing pair {image1['path']} and {image2['path']}: {str(e)}")
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.2f}")
    return accuracy

def test2(enhanced_images):
    correct = 0
    total = 0
    for enhanced_img in enhanced_images:
        try:
            result = DeepFace.verify(
                img1_path=enhanced_img["path"],
                img2_path=enhanced_img["original_pair"]["path"],
                model_name="VGG-Face",
                enforce_detection=False,
            )
            if result["verified"]:
                correct += 1
            total += 1
        except Exception as e:
            print(f"Error testing pair {enhanced_img['path']} and {enhanced_img['original_pair']['path']}: {str(e)}")

    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy for augmented images: {accuracy:.2f}")
    return accuracy

if __name__ == "__main__":
    data_dir = "lfw"
    qualified_images = loadAndFilter(data_dir, max_people=500)
    # 初始测试
    print("Testing original dataset accuracy...")
    test1(qualified_images)

    # 性别分类测试
    male_images, female_images = classifyByGender(qualified_images)
    print("Testing gender-separated datasets...")
    print("Testing Male dataset...")
    test1(male_images)
    print("Testing Female dataset...")
    test1(female_images)

    # 年龄分类测试
    age_groups = classifyByAge(qualified_images)
    print("Testing age-separated datasets...")
    for age_group, images in age_groups.items():
        print(f"Testing {age_group} dataset...")
        test1(images)

    # 国籍分类测试
    nationality_groups = classifyByNationality(qualified_images)
    print("Testing nationality-separated datasets...")
    for nationality, images in nationality_groups.items():
        print(f"Testing {nationality} dataset...")
        test1(images)

    # 数据增广测试
    augmented_dir = "augmented_lfw_images"
    dark_images, blur_images = augmentImages(qualified_images, augmented_dir)
    print("Testing darkened dataset accuracy...")
    test2(dark_images)
    print("Testing blurred dataset accuracy...")
    test2(blur_images)
