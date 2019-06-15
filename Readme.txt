Simplest way to run the project would be to open the following notebook file in juliabox

>WumpusWorldGold.ipynb

optionally, you can run the file locally using following command

>julia WumpusWorld.jl

Due to large computation time, the default world has been simplified to

3X3 world,
1 wumpus,
1 gold


(I have included the output for 4X4 world as well)

You can change the world size by supplying different parameter values in WumpusPOMDP Constructor
in cell 3 line 10.

There are also files (listed below) for world consisting of 1 wumpus, 1 pit and 1 gold . But it takes a lot of time to run. You can try it if you have high performing PC.

>WumpusWorldGoldPitBreeze.ipynb
>WumpusWorldGoldPitBreeze.jl

This file can be easily extended to contain another pit as well as take orientation into account. But the problem is runtime.

I have also included the 

pomdpx , policy and dot files for 3X3 

pomdx and dot files for 4X4 (policy file is more than 1.5 GB)

Please see sample_out.png to see one simulation into action.


 





