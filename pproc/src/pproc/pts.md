Tropical Cyclone tracks output
===

The TC tracker software generates tracks as ASCII files, under a folder hierarchy. These contain the date/time, value/lat/lon of minimum pressure, value/lat/lon of maximum wind, and wind radii of each quadrant for 3 wind thresholds.

File naming
---

Example folder hierarchy:

```
# tar file:
  2022062800/pf/034/00012022062800_034_000360
# a--------- b- c--     a--------- c-- d-----

# contents:
  00012022062800_034_000360_atl
#     a--------- c-- d----- e--
  00012022062800_034_000360_aus
  00012022062800_034_000360_cnp
  00012022062800_034_000360_enp
  00012022062800_034_000360_nin
  00012022062800_034_000360_sin
  00012022062800_034_000360_spc
  00012022062800_034_000360_wnp
```

Where:
- a) basetime or date/time of the model run (2022-06-28 00)
- b) ensemble (pf), control (cf) or high-resolution (oper) (ensemble)
- c) ensemble number, absent for cf and oper (034)
- d) total forecast length, 240h, 360h or 1080h (360h)
- e) basin indentifier (*f* latitudes values should be multiplied by -1):
  - atl: North Atlantic
  - aus: Northern Autralia *f*
  - cnp: Central North Pacific
  - enp: Northeast Pacific
  - nin: North Indian Ocean
  - sin: South Indian Ocean *f*
  - spc: South Pacific *f*
  - wnp: Northwest Pacific


File contents
---

Example contents of `00012022062800_034_000360_atl`:

```
00440 28/06/2022 M= 5 4 SNBR= 4
#                  a-        b-
00450 2022/06/28/00*5343425  39  994*5343372*00000000000034800237*00000000000000000000*00000000000000000000*
#     c------------ d------ e-- f--- g------ h------------------- i------------------- j-------------------
00460 2022/06/28/06*5463456  37  993*5143405*00000000000029000166*00000000000000000000*00000000000000000000*
00470 2022/06/28/18*5623495  29  997*5563418*00000000000000000000*00000000000000000000*00000000000000000000*
00480 2022/06/29/00*5793498  28  999*5623449*00000000000000000000*00000000000000000000*00000000000000000000*
00490 2022/06/29/06*5893480  28  998*5553427*00000000000000000000*00000000000000000000*00000000000000000000*
00494 TS
#     k-
```

Where:
- a) number of points of track (5)
- b) feature tracked (4th)
- c) date/time (2022-06-28 00)
- d) latitude/longitude of minimum pressure (53.4/342.5)
- e) maximum wind speed (39kt)
- f) minimum pressure (994hPa)
- g) latitude/longitude of maximum wind speed (53.4/337.2)
- h) 34kt wind radii, 4x (NE, SE, SW, NW quadrants) 5-digit values (NE=0m, SE=0m, SW=348m, NW=237m)
- i) 50kt wind radii ...
- j) 64kt wind radii ...
- k) strength category attribution (TS)
  - TD: wind < 34kt, tropical depression
  - TS: wind >= 34kt, tropical storm
  - HR1: wind >= 64kt, hurricane category 1
  - HR2: wind >= 83kt, hurricane category 2
  - HR3: wind >= 95kt, hurricane category 3
  - HR4: ... hurricane category 4
  - HR5: ... hurricane category 5


Filtering
---

For filtering wind, check if *any* of the start/end track segment points exceed a minimum.

