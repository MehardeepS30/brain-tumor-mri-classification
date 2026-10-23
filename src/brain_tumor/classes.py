"""The four classes, in the fixed order used for every label index."""

# Alphabetical, matching the Kaggle folder names.
CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)
