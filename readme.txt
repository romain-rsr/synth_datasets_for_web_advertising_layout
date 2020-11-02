----------------------------------------------------------------------------------------------------------------------------------------------------------
								     sum up
----------------------------------------------------------------------------------------------------------------------------------------------------------

This repository contains two synthetic datasets of 100,000 samples each and a small real cases dataset of 100 samples. These datasets allow for the training and the evaluation of learning models on generation of advertising layouts, even in the case where no real training samples are available on your side.

In our first dataset, virtual layouts are created through fictitious layout rules that exacerbate the layout complexity encountered in real web ads, and makes it possible to evaluate the capacity of any experimented model to integrate such complexity. In our second dataset virtual layouts have been generated through a number of realistic layout rules, which are identical to some specific layout rules in real ads. This second dataset allows any learning model to be both pre-trained and evaluated on this synthetic dataset then to be used on real ad layouts with good results, even in the case where no real ad layout samples are available for training.

Both synthetic datasets can be created, loaded, and displayed through the data_processing_tools jupyter notebook.

----------------------------------------------------------------------------------------------------------------------------------------------------------
					 	specific rules for the fictitious synthetic dataset
----------------------------------------------------------------------------------------------------------------------------------------------------------

Each trigram on the left represents a color combination
e.g. : BBB means that the elements colors of the layout are blue-blue-blue
e.g. : GGR means that the elements colors of the layout are green-green-red

BBB  => on left border
BGB  => last on bottom, penultimate few pixels above last
GGG  => on corners (all expected low left corner)
RRR  => all at mid line
RRB  => all at top line
BBG  => all at bottom line
GGR  => on the right
BRB  => diagonal going low right, without overlap
RBR  => diagonal going low left, without overlap
RGB  => diagonal with overlap, centered on first (red) elem

