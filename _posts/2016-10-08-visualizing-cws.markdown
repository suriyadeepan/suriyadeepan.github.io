---
layout: post
title: "Weather Report from Puducherry"
subtitle: "Visualizing data from Puducherry Community Weather Station"
tags: [ "visualization", "weather station", "community" ]
published: false
---

5 months ago, our team of hardware hackers build and deployed a [DIY standalone Weather station](http://www.instructables.com/id/DIY-Standalone-Weather-Station-Powered-by-Arduino/) powered by Open Hardware and Free Software, at Pondicherry Science Forum](https://www.google.co.in/maps/dir/''/pondicherry+science+forum/@11.9328189,79.7274214,12z/data=!4m8!4m7!1m0!1m5!1m1!1s0x3a536108247fffe3:0x72fe2d07a198b56f!2m2!1d79.797462!2d11.9328276). The data is being pushed to [opendata](http://opendata.fsftn.org/streams/ymkg3zJBGgF6vkZDw2M8I81JG9Q). Now we have a [need](https://discuss.fsftn.org/t/visualising-data-provided-by-community-weather-station-puducherry/808) to visualize the sensory data being recorded. This article is a documentation of my attempt to do just that.

There is an ocean of tools/frameworks for visualizing data. I do not want to drown in that. Not yet. My objective is to represent the data in visual format, as simple as possible, to get my feet wet. The code for processing the data is available at [fshm/community-ws-viz](https://gitlab.com/fshm/community-ws-viz).

**1. A look at the data**

![](/img/ws_head.png)

We have got wind speed and direction, humidity, pressure and temperature.

1. Wind direction : SW


## References

1. [Building a Low Cost Community Weather Station](https://discuss.fsftn.org/t/building-a-low-cost-community-weather-station/360) . 
