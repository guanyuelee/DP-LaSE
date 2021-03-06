# DP-LaSE
Implementation for the paper *Discovering Density-Preserving Latent Space Walks in GANs for Semantic Image Transformations* published in ACM Multimedia, 2021. For more details, please refer to my [personal website](https://guanyueli.com/publication/multimedia2021) or read the [PDF](https://1drv.ms/b/s!AqN-jN9xngyFohsv5_4BvANJPMcG). 

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="./docs/arch.png">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: black;
    padding: 2px;">
    Teaser: The flowchart of the DP-LaSE. 
    </div>
</center>

# Environment
I provide the conda environment in file *conda_env.yaml* if you are interested. 
```
conda env create -f conda_env.yaml
```

# Run the Code
All the relavant code is released, but they are quit messy right now. I will edit them after this tough PhD application cycle. 

# Some Results
I demonstrate some directions in BigGAN and StyleGAN. For dog in BigGAN, we mainly found some directions that cause position change, like left-right, zoom in-out and up-down. For human face in StyleGAN, various directions are found. If the intermediate layer we set to optimize the maximum variation is close to the input, the attribution we found are age, gender, etc. Otherwise, attributes like glasses and bear can be found. 
<table>
    <tr>
        <td>
            <center>
            <img src="https://guanyueli.com/images/papers/acmmm2021/dog-left-right.gif" width=300>
            <br>
            <div style="color:orange;
            display: inline-block;
            color: black;
            padding: 2px;">
            (a) Left-right
            </div>
            </center>
        </td>
        <td>
            <center>
            <img src="https://guanyueli.com/images/papers/acmmm2021/dog_zoom.gif" width=300>
            <br>
            <div style="color:orange;
            display: inline-block;
            color: black;
            padding: 2px;">
            (b) Zoom in - out
            </div>
            </center>
        </td>
        <td>
            <center>
            <img src="https://guanyueli.com/images/papers/acmmm2021/dog_up_down.gif" width=300>
            <br>
            <div style="color:orange;
            display: inline-block;
            color: black;
            padding: 2px;">
            (c) Up - down
            </div>
            </center>
        </td>
    </tr>
</table>
<center>
<div style="color:orange;
    display: inline-block;
    color: black;
    padding: 2px;">
    Figure 3. Some of the directions in BigGAN. 
    </div>
</center>

<table>
    <tr>
        <td>
            <center>
            <img src="https://guanyueli.com/images/papers/acmmm2021/bear.gif" width=300>
            <br>
            <div style="color:orange;
            display: inline-block;
            color: black;
            padding: 2px;">
            (a) Bear
            </div>
            </center>
        </td>
        <td>
            <center>
            <img src="https://guanyueli.com/images/papers/acmmm2021/hair.gif" width=300>
            <br>
            <div style="color:orange;
            display: inline-block;
            color: black;
            padding: 2px;">
            (b) Hair
            </div>
            </center>
        </td>
        <td>
            <center>
            <img src="https://guanyueli.com/images/papers/acmmm2021/gender.gif" width=300>
            <br>
            <div style="color:orange;
            display: inline-block;
            color: black;
            padding: 2px;">
            (d) Gender
            </div>
            </center>
        </td>
        <td>
            <center>
            <img src="https://guanyueli.com/images/papers/acmmm2021/glass.gif" width=300>
            <br>
            <div style="color:orange;
            display: inline-block;
            color: black;
            padding: 2px;">
            (c) Glasses
            </div>
            </center>
        </td>
    </tr>
</table>
<center>
<div style="color:orange;
    display: inline-block;
    color: black;
    padding: 2px;">
    Figure 1. Some of the directions in PGGAN. (Github cannot show gif file exceeding 5 MB, please refer to my blog  https://guanyueli.com/publication/multimedia2021. )
    </div>
</center>