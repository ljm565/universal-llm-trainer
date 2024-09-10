# Data Preparation
To perform LLM instruction-tuning, please prepare the data as follows.

## Data
### 1. Data Path
You have to set data path as follows:
```
${DATA_DIR}                         
     └── ${DATA_NAME}               <- Name of data
         └── ${DATA_NAME}.pkl       <- Dataset pickle file (Must be saved as "DATA_NAME")
```
For example, you can save the data in the path shown above and use it in the config file in the `config` folder.
```bash
# Data path
# DATA_DIR=data, DATA_NAME=ai2_arc
data/ai2_arc/ai2_arc.pkl
```


### 2. Data Construction
For training, you need to configure the pickle data as shown below.
```json
{
   "alpaca_style":{
      "train":[
         {
            "input":[],
            "instruction":["Please recommend summer vacation spots.", "I like mountains."],
            "output":["In summer, I feel like I could choose between the mountains and the sea.", "If you like mountains, how about the Dolomites?"]
         },
         ...
      ],
      "validation":[
         {
            "input":["나는 너를 사랑해"],
            "instruction":["Please translate the above sentence in English."],
            "output":["Here is the results: I love you."]
         },
         ...
      ]
   }
}
```
<br>

#### Cautions
* The top key must be set as `alpaca_style` (To be revised).
* The subkeys of `alpaca_style` must be `train` and `validation`, and each should follow the structure `List[Dict]`.
* Each element of `train` (or `validation`) is a dictionary composed of the keys: `input`, `instruction`, and `output`.
  1. `input`, `instruction`, and `output` must be lists.
  2. `instruction` and `output` can have more than one element. When there is more than one element, you can enable the multi-turn option during training for multi-turn learning.
  3. `input` is optional. It can either be present or absent.