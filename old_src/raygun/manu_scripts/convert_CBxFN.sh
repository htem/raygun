export DISPLAY=:1
Xvfb $DISPLAY -auth /dev/null &
(
# the '(' starts a new sub shell. In this sub shell we start the worker processes:

~/fiji -macro /n/groups/htem/users/jlr54/raygun/manu_scripts/convert_CBxFN.ijm # running the actual ijm script

wait # waits until all 'program' processes are finished
# this wait sees only the 'program' processes, not the Xvfb process
)