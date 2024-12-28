from deepface import DeepFace
result = DeepFace.verify(
    img1_path="3.jpg",
    img2_path="4.jpg",
)

print("Result: ", result)