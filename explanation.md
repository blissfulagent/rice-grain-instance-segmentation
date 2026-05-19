# Explanation Document

## Approach

For this task, I built a classical computer vision pipeline to generate a stylized instance-segmentation output for rice grains. Since the input image has a dark background and relatively brighter rice grains, I first focused on separating the foreground grains from the background using grayscale conversion, smoothing, and intensity-based thresholding.

After obtaining the foreground mask, I applied a distance transform to identify high-confidence center regions inside the rice grains. Local maxima from the distance map were used as seed points, since these points usually lie near the center of individual grains.

Initially, I experimented with a watershed-based approach for separating touching grains. However, the output became noisy and irregular because several grains were overlapping, low-contrast, or softly connected. Since the expected output is a stylized colored segmentation rather than an exact pixel-level mask, I used the detected seed points to render smooth rice-shaped colored regions. Each seed point is treated as one grain instance, and a local orientation estimate is used to draw an ellipse-like shape around it.

The final output keeps the background black and assigns distinct colors to the detected grain instances.

## Key Observations

- The rice grains are generally brighter than the background, so intensity-based foreground extraction works reasonably well.
- The main difficulty is not background removal, but separating individual grains where they touch or overlap.
- A clean foreground mask is very important. If the mask contains background noise, later stages also become noisy.
- Distance transform is useful because it highlights the inner regions of grains and helps locate seed points.
- Direct watershed segmentation produced jagged and fragmented regions in some cases.
- A seed-based ellipse rendering approach gave a cleaner visual output closer to the provided reference image.

## Challenges Encountered

- Some rice grains have weak contrast and partially blend with nearby grains.
- Several grains are touching or overlapping, making exact individual separation difficult.
- Adaptive thresholding initially introduced extra background noise, so I shifted to a simpler threshold-based foreground mask.
- Watershed segmentation was tested, but it produced noisy boundaries and several small unwanted fragments.
- Parameter tuning was required for foreground threshold, seed spacing, and ellipse size to get a visually balanced output.
- There is a tradeoff between detecting more grains and avoiding false detections in noisy regions.

## Potential Future Improvements

- Use SAM/SAM2 or another pretrained segmentation model to generate stronger object masks.
- Fine-tune an instance segmentation model if more annotated rice-grain images are available.
- Improve handling of highly overlapping grains where one seed may not be enough.
- Add automatic parameter selection so the method adapts better to different lighting conditions.
- Combine the current classical pipeline with a learned segmentation/refinement model for more accurate boundaries.