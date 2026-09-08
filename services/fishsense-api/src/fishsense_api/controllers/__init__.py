"""Controllers package for FishSense API."""

import fishsense_api.controllers.calibration_candidate_controller
import fishsense_api.controllers.camera_controller

# ORDER IS LOAD-BEARING: the cohort selectors register `/dives/select-next/...`
# and must precede `dive_controller`'s `/dives/{dive_id}`. FastAPI matches in
# declaration order, which across modules is import order here — get it wrong
# and every selector request tries to coerce "select-next" into an int path
# param and 422s. Guarded by test_dive_route_disambiguation.py.
import fishsense_api.controllers.dive_cohort_controller

# NOT alphabetical, deliberately: placed with `dive_cohort_controller`, before
# `dive_controller`, because that is the convention for every module holding
# `/dives/...` collection routes.
#
# Measured, so the comment does not overstate it: the catch-all compiles to
# `^/api/v1/dives/(?P<dive_id>[^/]+)$` — anchored, one segment, no trailing
# slash — so it cannot shadow these routes, which have two segments and a
# trailing slash. Reordering this import does *not* break them today, and
# `test_dive_route_disambiguation` passes either way. The ordering is kept so
# that a selector added here later as a bare single segment (the shape
# `/dives/needing-*-population/` nearly is) stays safe by construction.
import fishsense_api.controllers.dive_prediction_cohort_controller
import fishsense_api.controllers.dive_controller
import fishsense_api.controllers.dive_slate_controller
import fishsense_api.controllers.fish_controller
import fishsense_api.controllers.head_tail_prediction_controller
import fishsense_api.controllers.image_controller
import fishsense_api.controllers.label_controller
import fishsense_api.controllers.laser_depth_controller
import fishsense_api.controllers.laser_prediction_controller
import fishsense_api.controllers.slate_prediction_controller
import fishsense_api.controllers.user_controller
