# Target: CelebA Encoder

One of the most important targets. Trained on a different dataset than our generator (see 0_2_celebA_autoencoder for more on the dataset). 

The identity classification vector is large, containing one element/one class for each of the 10177 depicted identities. We trained a ReNet18 with CELoss for this for 200 epochs. The attribute vector is made up of 40 **binary** attributes, namely:

    (f) '5_o_Clock_Shadow'
    (f) 'Arched_Eyebrows'
    (f) 'Attractive'
    (f) 'Bags_Under_Eyes'
    (h) 'Bald'
    (h) 'Bangs'
    (f) 'Big_Lips'
    (f) 'Big_Nose'
    (h) 'Black_Hair'
    (h) 'Blond_Hair'
    'Blurry'
    (h) 'Brown_Hair'
    (f) 'Bushy_Eyebrows'
    (f) 'Chubby'
    (f) 'Double_Chin'
    (f) 'Eyeglasses'
    (f) 'Goatee'
    (h) 'Gray_Hair'
    (f) 'Heavy_Makeup'
    (f) 'High_Cheekbones'
    (f) 'Male'
    (e) 'Mouth_Slightly_Open'
    (f) 'Mustache'
    (f) 'Narrow_Eyes'
    (f) 'No_Beard'
    (f) 'Oval_Face'
    (f) 'Pale_Skin'
    (f) 'Pointy_Nose'
    (f) 'Receding_Hairline'
    (f) 'Rosy_Cheeks'
    (f) 'Sideburns'
    (e) 'Smiling'
    (h) 'Straight_Hair'
    (h) 'Wavy_Hair'
    (a) 'Wearing_Earrings'
    (a) 'Wearing_Hat'
    (a) 'Wearing_Lipstick'
    (a) 'Wearing_Necklace'
    (a) 'Wearing_Necktie'
    (f) 'Young'

The optimal attributes are binary, but I am using the direct output of the encoder. Of the 40 attributes, 
* (f) 24 are describing facial features. Some of them are highly correlated (e.g. "no_beard" and "mustache"), and most very subjective (most notably "attractive", "young"), which is however not necessarily bad for the reconstruction. 
* (h) 8 are for the hair, four for the color (brown, black, blond, gray, **not red**), four for the "shape" (bald, bangs, straight, wavy). Again, a lot of correlation. For example, there can always only be one of the four colors.
* (e) 2 for expression
* (a) 5 for accessories
* and, last but not least, one is "blurry".

According to the paper, the attributes are found by making "a professional labeling company" classify those features for "the 8000 identities". There is no further information given about the company, or about who it was that considered "attractive" to be a binary attribute that should be classified by this mystery company. 

I trained a ResNet18 for both, CELoss for the identity and MSE for the attributes.