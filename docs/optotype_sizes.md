# Snellen Chart & Optotype Sizes

This document provides a guide to understanding optotype sizes based on the Snellen chart provided in `Sample/snellen-white-on-black-778x1024.jpg`.

## Understanding Snellen Frations

A Snellen fraction (e.g., 20/20 or 6/6) expresses visual acuity:
- **Numerator**: The distance at which the test is performed (usually 20 feet or 6 metres).
- **Denominator**: The distance at which a person with "normal" vision can read the same line.

## Visual Acuity Levels

Based on the chart image, here are the corresponding Metre and Feet acuities:

| Metric (Metres) | Imperial (Feet) | Notes |
| :--- | :--- | :--- |
| 6/60 | 20/200 | Legally Blind (top letter 'E') |
| 6/30 | 20/100 | |
| 6/20 | 20/70 | |
| 6/15 | 20/50 | |
| 6/12 | 20/40 | |
| 6/9 | 20/30 | |
| --- | --- | **RED LINE** |
| 6/7.5 | 20/25 | |
| 6/6 | 20/20 | "Normal" Vision |
| --- | --- | **GREEN LINE** |
| 6/5 | 20/16 | Better than normal |
| 6/4 | 20/13 | |
| 6/3 | 20/10 | |

## Optotype Size Calculation

The standard Snellen optotype (like the letter 'E') is designed to subtend an angle of **5 minutes of arc** (5') at the specified denominator distance. Each "detail" of the letter (e.g., the thickness of the bars of the 'E') subtends **1 minute of arc** (1').

### Formula

The height ($H$) of an optotype can be calculated using the tangent of the subtended angle:

$$H = d \times \tan(\theta)$$

Where:
- $d$ = Reference distance (the denominator, e.g., 60m for the 6/60 line).
- $\theta$ = 5 minutes of arc = $5/60$ degrees $\approx 0.0833^\circ$.

At small angles, $\tan(\theta) \approx \theta$ (in radians).
$5' = \frac{5}{60 \times 180} \times \pi$ radians $\approx 0.001454$ radians.

**Simplified Formula for 6m / 20ft testing distance:**
The size of the letter on the screen/chart depends on the denominator $D$ of the Snellen fraction $6/D$.

$$Height (mm) \approx D \times 1.454 \times \frac{Testing Distance}{Reference Distance}$$

For a standard **6 metre** testing distance:
$$Height (mm) = D \times \tan(5') \times \frac{6000}{D} = 6000 \times \tan(5')$$
Wait, the height of the letter that subtends 5' at distance $D$ is $D \tan(5')$. 
So at 6 meters, the letter for $6/D$ must be the same size that the $6/6$ letter would be at distance $D \times (6/6)$.

Actually, the standard height for **6/6 (20/20)** at **6 metres** is:
$$H = 6000 \text{ mm} \times \tan(5') \approx 8.73 \text{ mm}$$

### Predicted Heights at 6m (20ft) Distance

| Snellen Line | Denominator ($D$) | Letter Height (mm) | Letter Height (inches) |
| :--- | :--- | :--- | :--- |
| 6/60 (20/200) | 60 | 87.27 | 3.436 |
| 6/30 (20/100) | 30 | 43.63 | 1.718 |
| 6/20 (20/70) | 20 | 29.09 | 1.145 |
| 6/15 (20/50) | 15 | 21.82 | 0.859 |
| 6/12 (20/40) | 12 | 17.45 | 0.687 |
| 6/9 (20/30) | 9 | 13.09 | 0.515 |
| 6/6 (20/20) | 6 | 8.73 | 0.344 |
| 6/3 (20/10) | 3 | 4.36 | 0.172 |

## Color Bar Significance

- **Red Line**: Often indicates the 6/9 (20/30) level. In some clinical contexts, failing to read below the red line may indicate a need for further investigation or meeting specific vision requirements (e.g., for driving).
- **Green Line**: Typically indicates the 6/6 (20/20) "normal" vision threshold. Successfully reading the line above the green line (6/6) is the standard target.
