---
layout: page
title: "Shomer Shabbos"
subtitle: "Day of rest"
---

Okay. Let us see if SMIL works in static markdown page.

![](/img/smil/dynamic-rect.svg)

Whoa! So, it is working. The text doesn't show up though. I wonder why?

Here is the source of the svg.

```xml
<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <title>Simple animate example</title>
  <desc>Rectangle shape will change</desc>
  <rect x="50" y="50" width="10" height="100" style="fill:#CCCCFF;stroke:#000099">
    <animate attributeName="width"  from="10px"  to="100px" 
	     begin="0s" dur="10s" />
    <animate attributeName="height" from="100px" to="10px"
	     begin="0s" dur="10s" />
  </rect>
  <text x="50" y="170" style="stroke:#000099;fill:#000099;font-size:14;">
  Hello. Admire the dynamic rectangle</text>
</svg>
```
