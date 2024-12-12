import tensorflow_datasets as tfds
import os

dataset_dir = 'H:/MGSTR/imagenet'  # directory where you downloaded the tar files to
temp_dir = 'H:/MGSTR/imagenet/temp'  # a temporary directory where the data will be stored intermediately

download_config = tfds.download.DownloadConfig(
    extract_dir=os.path.join(temp_dir, 'extracted'),
    manual_dir=dataset_dir
)

tfds.builder("imagenet2012").download_and_prepare(download_config=download_config)
