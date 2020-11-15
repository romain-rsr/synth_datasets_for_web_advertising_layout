-----------------------------------------------------------------------------------------------------------------
                                                sum up
-----------------------------------------------------------------------------------------------------------------

This repository contains two synthetic datasets of 100,000 samples each and a small real cases dataset of 100 samples. 
These datasets allow for the training and the evaluation of learning models on generation of advertising layouts, even 
in the case where no real training samples are available on your side.

In our first dataset, virtual layouts are created through fictitious layout rules that exacerbate the layout complexity 
encountered in real web ads, and makes it possible to evaluate the capacity of any experimented model to integrate such 
complexity. In our second dataset virtual layouts have been generated through a number of realistic layout rules, which 
are identical to some specific layout rules in real ads. This second dataset allows any learning model to be both 
pre-trained and evaluated on this synthetic dataset then to be used on real ad layouts with good results, even in the 
case where no real ad layout samples are available for training.

Both synthetic datasets can be created, loaded, and displayed through the data_processing_tools jupyter notebook.

-----------------------------------------------------------------------------------------------------------------
	                                    list of datasets
-----------------------------------------------------------------------------------------------------------------

synth1 : first synthetic dataset of 100K samples, created through explicit, fictitious layout rules
synth2 : second synthetic dataset of 100K samples, created through explicit, realistic layout rules
real : real cases dataset of few (93) samples, created by designer through intuitive, implicit layout rules 

-----------------------------------------------------------------------------------------------------------------
	        specific rules for the fictitious synthetic dataset (dataset "synth1")
-----------------------------------------------------------------------------------------------------------------

In this dataset, most of the layouts have been created randomly within general constraints, 
while a minority of other layouts in this dataset have been created through specific rules. 

Each specific rule is applied to layouts where the colors of the elements match a specific sequence.
To get the color sequence of a layout, we check the color of each element in the reading order of the elements,
which is randomly set in this first synthetic dataset.

# general constraints

- elements must not overlap and a minimum space must be kept empty between them
- elements must not exceed borders

# specific constraints

- bbb  => the left border of each element must stick to the left border of the screen
- bgb  => last element stick on the bottom of the screen, penultimate stick a few pixels above last element
- ggg  => elements stick on corners (all expected low left corner)
- rrr  => all elements are placed at the middle of the screen, with equal dimensions and distance from each other
- rrb  => all elements are placed at the top of the screen, with equal dimensions and distance from each other
- bbg  => all elements are placed at the bottom of the screen, with equal dimensions and distance from each other
- ggr  => all elements are placed at the bottom of the screen, with equal dimensions and distance from each other
- brb  => elements form a diagonal going in the low right direction, without overlap
- rgb  => elements form a diagonal going in the low right direction, with specific overlap
- rbr  => elements form a diagonal going in the low left direction, without overlap

NB : each trigram on the left represents a color combination
e.g. : bbb means that the elements colors of the layout are : blue-blue-blue
e.g. : ggr means that the elements colors of the layout are : green-green-red
