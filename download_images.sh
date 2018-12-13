#!/usr/bin/env bash

#Creative-commons licensed flower photos curated by TensorFlow
curl -O http://download.tensorflow.org/example_images/flower_photos.tgz
tar xzf flower_photos.tgz
mkdir flower_photos/sample

#Let’s copy some photos from the tulips and daisy folders to create a small sample of the photos.
curl -O http://download.tensorflow.org/example_images/flower_photos.tgz
tar xzf flower_photos.tgz
mkdir flower_photos/sample