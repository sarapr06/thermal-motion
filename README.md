# Thermal Motion & Boltzmann's Constant

This repository contains the data analysis scripts, raw tracking data, and resulting figures for an experimental physics lab investigating the Brownian motion of fluorescent microscopic beads suspended in water. 

The primary objective of this project is to calculate Boltzmann's constant ($k$) using two distinct statistical methods: 1D mean squared displacement (MSD) over varying time lags, and a maximum likelihood estimate (MLE) of 2D radial step distributions.

## Repository Structure

```text
thermal_motion/
│
├── tm_aarya_rec1/               # Directory containing raw .txt tracking data(X, Y coordinates), 2 in each (rest are images of particles)
│
├── thermalmotionpt1.py          # Part 1 Script: MSD vs. Time Lag analysis & linear regression
├── thermalmotionpt2.py          # Part 2 Script: 2D Step distribution, Curve Fit, and MLE
│
├── msd_results.csv              # Output data from Part 1 containing calculated MSDs
├── msd_plot.png                 # Output plot: 1D MSD vs. Time Lag
├── rayleigh_histogram.png       # Output plot: Step length histogram with Rayleigh fits
│
├── apparatus.png                # Lab report asset: Photo of the experimental setup
├── thermal motion.png           # Lab report asset: Diagram/supplementary image
├── IMG_5566.HEIC                # Lab report asset: Raw setup photo
├── IMG_5568.HEIC                # Lab report asset: Raw setup photo
│
├── ThermalMotion.pdf            # Original laboratory manual and project guidelines
└── README.md                    # Project documentation