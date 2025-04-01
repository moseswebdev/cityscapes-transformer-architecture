class CityscapesDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, processor=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = os.listdir(image_dir)
        self.transform = transform
        self.processor = processor

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.image_filenames[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.array(mask, dtype=np.uint8)  # Ensure NumPy format

        # 🚀 Fix: Ensure masks are within range
        mask[mask >= 19] = 18  # Limit mask values to valid classes (0-18)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        mask = torch.tensor(mask, dtype=torch.long)  # Convert back to tensor

        if self.processor:
            inputs = self.processor(image, return_tensors="pt")
            image = inputs['pixel_values'].squeeze(0)

        return image, mask
