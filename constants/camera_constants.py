N_AVG = 5            # frames in the rolling average
CROP = 200           # ignore this many pixels at each edge of the frame
BAND_UM = 330        # half-height of the analysed band (µm); must exceed one groove pitch
MIN_COVERAGE = 0.5   # fraction of the visible arc a groove must span to be measured

UM_PER_PX = 2.74 / 0.5      # pixel size on the disc (µm)
SECTION_PX = 40            # length of each cross-section along the grooves (px)
CENTRE = (13010, -1478)      # disc centre in pixels (x, y) - found by least-squares intersection of groove normals
