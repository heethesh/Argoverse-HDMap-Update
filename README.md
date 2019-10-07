# Argoverse HDMap Updator
CMU 16-822 Geometry Vision Project - Updating HD map information with data from smartphones

## About
High Definition (HD) maps are an important commodity for modern self-driving car companies as they are useful to precisely localize in the city. Modern datasets such as the Argoverse dataset [1], annotate the driveable areas in the map data. However, the driveable areas need to be constantly updated with changing conditions of a city. Construction zones can pop up, accidents can occur, or weather can affect road conditions — all of which can cause lane closures or other obstructions to a routine drive. The current solution is to collect similar high definition data using special purpose vehicles regularly to update the changes. However, this solution is costly, especially for large-scale mapping in multiple cities. Instead, we propose a low cost  solution which aims to update the driveable area of HD maps using crowd-sourced images taken from a smartphone.

## References
```
@INPROCEEDINGS {Argoverse,
  author = {Ming-Fang Chang and John W Lambert and Patsorn Sangkloy and Jagjeet Singh
       and Slawomir Bak and Andrew Hartnett and De Wang and Peter Carr
       and Simon Lucey and Deva Ramanan and James Hays},
  title = {Argoverse: 3D Tracking and Forecasting with Rich Maps},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2019}
}
```
