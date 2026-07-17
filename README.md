# Deep Learning for Geospatial Analysis: Automating Building Recognition Across Prince Edward Island (1968)

## Project Objective
This project aimed to achieve clear segmentation and mapping of historic buildings (barns, houses, and other structures such as stores, sheds, and gas stations) using historical aerial imagery of Prince Edward Island from 1968.

## Initial Challenges
1.  **Data Complexity:** Dealing with a massive, black-and-white historical map required specialized handling and custom dataset creation.
2.  **Computational Limitations:** The deep learning capabilities built into ArcGIS were not performant enough on available work computers, necessitating an external deep learning solution.

## Technical Journey and Methodology

### 1. Data Preparation and Custom Dataset Development
*   **Manual Segmentation:** Approximately 14,000 building entries were manually segmented across Prince County, forming the basis of the training data.
*   **PyTorch Dataset Implementation:** A robust `HistoricBuildingDataset` was developed using PyTorch. This custom data loader was designed to:
    *   Handle complex `.tif` image formats and their associated instance segmentation masks.
    *   Perform dynamic image and mask loading, including processing of instance segmentation masks.
    *   Enable efficient batching for model training.

### 2. Model Selection and Architecture
*   **Mask R-CNN:** A pre-trained `Mask R-CNN` model, equipped with a ResNet-50-FPN backbone, was selected and fine-tuned for the task. This architecture was chosen for its proven capability in high-accuracy object detection and instance segmentation across various object classes.

### 3. Model Training
*   **GPU-Accelerated Training:** The model underwent 10 epochs of training on Google Colab, leveraging its GPU resources for faster computation.
*   **Performance:** Training was conducted on 6,044 images, resulting in a validation loss of approximately 0.31, indicating strong model performance.

### 4. Memory-Efficient Geospatial Inference Pipeline
To process hundreds of thousands of high-resolution `.tif` files across entire counties (Prince and Kings Counties), a specialized memory-efficient pipeline was developed:
*   **Intelligent Tiling:** Massive county-level TIFF files were broken down into smaller, overlapping tiles. This strategy managed memory effectively while ensuring full coverage of the map areas.
*   **Filtering for Relevance:** To optimize processing time, irrelevant tiles were filtered out. This included:
    *   Discarding empty ocean areas (NoData flags).
    *   Removing black clusters.
    *   Filtering tiles with low variance, indicating lack of meaningful visual information.
*   **Batch Processing & Checkpointing:** Model inference on these tiles was optimized through batch processing. A checkpointing mechanism was implemented to save processed progress incrementally, allowing for recovery and resumption in case of system interruptions.

### 5. GeoJSON Conversion and Spatial Intelligence
*   **Output Format:** The model's pixel-level predictions were converted into the industry-standard `GeoJSON` format, crucial for integration with ArcGIS.
*   **Coordinate Transformation:** Pixel coordinates from the `.tif` images were accurately transformed to WGS84 (latitude/longitude), ensuring global compatibility.
*   **Area Calculation:** The real-world area of detected buildings was precisely calculated in square meters using the original Coordinate Reference System (CRS) and embedded into the GeoJSON properties.

### 6. Advanced Duplicate Handling
To address the issue of multiple overlapping polygons generated for a single building due to tiled inference, a multi-stage duplicate handling approach was implemented:
*   **Prince County (Spatial Non-Maximum Suppression):** Initially, a spatial NMS-like technique was employed to remove duplicate detections based on Intersection-over-Union (IoU) metrics.
*   **Kings County (Refined GeoPandas Dissolve/Explode):** A more robust method using `GeoPandas` was developed, involving:
    *   **Micro-Buffering:** Expanding polygons by 0.5 meters to bridge microscopic gaps between tiles.
    *   **Same-Class Dissolve:** Merging touching polygons of the same building class into single, cohesive geometries.
    *   **Explode:** Breaking down merged multipolygons back into individual, distinct building shapes.
    *   **Restore Original Dimensions:** Shrinking polygons by 0.5 meters to revert to their true ground size.
    *   **Cross-Class Overlap Filtering:** A final filter was applied to remove cross-class overlaps. If two or more polygons (even of different classes) overlapped by a 30% threshold, the detection with the highest confidence score was retained, drastically reducing redundant shapes. This refined process resulted in over 9,300 unique, perfectly merged properties.

## Final Review and Project Outcome
*   **ArcGIS Integration:** The generated GeoJSON files were imported back into ArcGIS for manual correction, inspection, and further analysis.
*   **Workflow Value:** This project demonstrates a powerful workflow for automating the creation of valuable geospatial datasets from historical aerial imagery.
*   **Historical Insights:** The generated dataset will be instrumental in uncovering historical insights into Prince Edward Island from 1968, including identifying housing clusters, land distribution patterns, and understanding the broader geographical state of the island.

## Final result example implemented into arcGIS

![Detection 1](images/result1.png)

![Detection 2](images/result2.png)

## Technologies Utilized
Key open-source libraries critical to this project include PyTorch, torchvision, rasterio, shapely, geojson, geopandas, and numpy.
