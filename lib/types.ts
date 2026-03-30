export type MnistStats = {
  dataset_name: string;
  generated_at: string;
  labels: number[];
  summary: {
    classes: number;
    image_size: [number, number];
    flattened_features: number;
    pixel_value_range: [number, number];
    train_samples: number;
    test_samples: number;
  };
  class_counts: {
    train: number[];
    test: number[];
  };
  preview_file: string;
};

export type PredictionResponse = {
  predicted_digit: number;
  probabilities: number[];
};

export type ImageDataLike = {
  width: number;
  height: number;
  data: ArrayLike<number>;
};

