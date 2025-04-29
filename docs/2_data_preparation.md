# Data Preparation
To perform LLM instruction-tuning, please prepare the data as follows.

&nbsp;

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

&nbsp;

### 2. Data Construction
For training, you need to configure the pickle data as shown below.
```json
{
   "${KEY1}":{
      "train":[
         {
            "input": [],
            "instruction": ["Please recommend summer vacation spots.", "I like mountains."],
            "output": ["In summer, I feel like I could choose between the mountains and the sea.", "If you like mountains, how about the Dolomites?"]
         },
         ...
      ],
      "validation":[
         {
            "input": ["You are an language translator."],
            "instruction": ["Please translate the below sentence in English.\n\n나는 너를 사랑해."],
            "output": ["Here is the results: I love you."]
         },
         ...
      ]
   },
   "${KEY2}":{
      "train":[
         {
            "input": ["You ara mathematician."],
            "instruction": ["Calculate the below equation.\n\nWhat is the answer of `13 + 3 * 4`?"],
            "output": ["The answer is 25. When calculating, the multiplication operator must be performed first."]
         },
         ...
      ],
      "validation":[
         {
            "input": ["You are a reansing model."],
            "instruction": ["How many apples are remain?\n\nThere are five apples. My friends ate 4 apples and my mom bought 3 more apples."],
            "output": ["There are four apples in total."]
         },
         ...
      ]
   }
}
```
<br>

#### Cautions
* `${KEY}` can be any value, and can have multiple keys.
* The subkeys of `alpaca_style` must be `train` and `validation`, and each should follow the structure `List[Dict]`.
* Each element of `train` (or `validation`) is a dictionary composed of the keys: `input`, `instruction`, and `output`.
  1. `input`, `instruction`, and `output` must be lists.
  2. `instruction` and `output` can have more than one element. When there is more than one element, you can enable the multi-turn option during training for multi-turn learning.
  3. `input` is similar to system message (please refer to some templates in tempalates folder), so it is optional. It can either be present or absent.