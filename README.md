# Tumor detection
## Tumor classification:
* normal brain without tumor
* Astrocitoma
* Carcinoma
* Ependimoma
* Gandlioglioma
* Germinoma 
* Glioblastoma 
* Granuloma 
* Meduloblastoma
* Meningioma 
* Neurocitoma 
* Oligodendroglioma 
* Papiloma 
* Schwannoma
* Tuberculoma
<br>
## Model summary:
<table>
    <tr>
        <th>Layer Number</th>
        <th>Layer Name</th>
        <th>Output Shape</th>
        <th>Number of Parameters</th>
    </tr>
    <tr>
        <td>1</td>
        <td>Conv2d (relu)</td>
        <td>(None, 296, 296, 32)</td>
        <td>2,432</td>
    </tr>
    <tr>
        <td>2</td>
        <td>MaxPooling2D</td>
        <td>(None, 148, 148, 32)</td>
        <td>0</td>
    </tr>
    <tr>
        <td>3</td>
        <td>Conv2D (relu)</td>
        <td>(None, 146, 146, 64)</td>
        <td>18,496</td>
    </tr>
    <tr>
        <td>4</td>
        <td>MaxPooling2D</td>
        <td>(None, 73, 73, 64)</td>
        <td>0</td>
    </tr>
    <tr>
        <td>5</td>
        <td>Flatten</td>
        <td>(None, 341056)</td>
        <td>0</td>
    </tr>
    <tr>
        <td>6</td>
        <td>Dense (leaky_relu)</td>
        <td>(None, 64)</td>
        <td>21,827,648</td>
    </tr>
    <tr>
        <td>7</td>
        <td>Dense (softmax)</td>
        <td>(None, 15)</td>
        <td>975</td>
    </tr>
</table>

### Params summary:
* Total params: 65,548,655 (250.05 MB)<br>
* Trainable params: 21,849,551 (83.35 MB)<br>
* Optimizer params: 43,699,104 (166.70 MB)<br>

### Compiler Arguments:
* optimizer: adam
* loss: categorical crossentropy


