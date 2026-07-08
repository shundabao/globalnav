# GLOBALNAV 外部 OD 原生标注子集

这些文件由原始测试集的人工标注自动筛选得到。筛选过程不使用 LLM，不手工指定
样本 ID，也不生成或改写 gold label。`raw` 字段保留了支撑筛选结论的原始记录。

## 推荐的主实验数据

| 数据集 | 明确起终点 | 原生多轮澄清 |
|---|---:|---:|
| MultiWOZ 2.4 | 225 | 156 |
| Schema-Guided Dialogue | 332 | 401 |
| ATIS | 609 | 0 |
| 合计 | 1,166 | 557 |

文件：

- `multiwoz24_explicit_od.jsonl`
- `multiwoz24_clarification.jsonl`
- `sgd_explicit_od.jsonl`
- `sgd_clarification.jsonl`
- `atis_explicit_single_pair.jsonl`
- `atis_multi_endpoint_stress.jsonl`（82 条，不计入主实验）

## 自动筛选规则

### MultiWOZ 2.4

明确 OD：

1. 测试集同一个用户 turn 的原始 `turn_label` 同时包含
   `departure` 和 `destination`；
2. 两个 gold value 都逐字出现在该用户 utterance 中（仅做小写、标点和空格归一化）。

第 2 条会排除依赖上文指代的样本。原始标注候选有 360 条，适合单轮直接输入
GLOBALNAV 的严格子集有 225 条。

澄清：

1. 当前用户 turn 只给出一个端点，原始 `belief_state` 中另一个端点为空；
2. 下一系统 turn 的原始 dialogue act 是对缺失端点的 `Request`；
3. 紧接着的用户回答在原始 `turn_label` 中补齐该端点，最终 gold 取回答
   turn 的完整 `belief_state`，因此也保留用户同时作出的端点修改；
4. 两个端点值分别逐字出现在相应用户 utterance 中。

满足前三条的原始标注候选有 190 条；加入第 4 条、采用回答后的最终 state，
并排除起点等于终点的退化请求后，严格子集有 156 条。

### Schema-Guided Dialogue

只扫描测试集的 `Buses_3`、`Flights_4` 和 `Trains_1` 服务。

明确 OD 要求同一用户 turn 对两个端点都有原始 `INFORM` action 和原始字符
span。澄清样本要求缺失端点不在当前 state 中，下一系统 turn 对它有原始
`REQUEST` action，并且紧接着的用户 turn 有原始 `INFORM` action 和字符 span。
最终 gold 使用回答 turn 的 endpoint action 覆盖初始值，以保留用户在回答中同时
作出的修改。

### ATIS

只读取测试集原始 BIO NER 标签。主子集要求一句话中恰好有一个 `fromloc.*`
span 和一个 `toloc.*` span，共 609 条，其中 590 条是 city-to-city。

另有 82 条包含多个 from/to 标注 span，放在 `atis_multi_endpoint_stress.jsonl`
中，不与单一 OD 主实验混算。ATIS 是单轮语料，没有可验证的系统追问和用户回答，
因此澄清样本数是 0，不能从中人工构造。

## 原始样例

MultiWOZ 2.4 明确 OD（`MUL0671.json`, turn 1）：

```text
usr: i am going to cambridge from birmingham new street .
turn_label:
  train-departure = birmingham new street
  train-destination = cambridge
```

MultiWOZ 2.4 原生澄清（`MUL1575.json`, turn 3）：

```text
USER: thanks . can you help me find a train , too ? i want to leave cambridge some time after 12:15 .
SYSTEM: and where would you like your train to take you ?
USER: i need the train should go to peterborough and it should leave on saturday .
gold: origin = cambridge; destination = peterborough
```

SGD 明确 OD（`2_00091`, turn 2, `Flights_4`）：

```text
USER: I need a flight on the 11th of this month from SD to Ciudad de Mexico.
origin_airport: surface = SD; canonical = San Diego
destination_airport: surface = Ciudad de Mexico; canonical = Mexico City
```

SGD 原生澄清（`2_00093`, turn 0, `Flights_4`）：

```text
USER: I would like to book a one way flight to Toronto, Ontario.
SYSTEM: When do you want to travel and where are you departing from?
USER: I will be departing from Atlanta on the 7th of March.
gold: origin = Atlanta; destination = Toronto
```

ATIS（row 0）：

```text
text: i would like to find a flight from charlotte to las vegas that makes a stop in st. louis
BIO spans:
  fromloc.city_name = charlotte
  toloc.city_name = las vegas
  stoploc.city_name = st. louis
```

## 来源与复现

- MultiWOZ 2.4: `smartyfh/MultiWOZ2.4`, commit
  `6807c1d85f547fcaae10494d26991d2d37c90a63`
- Schema-Guided Dialogue: `google-research-datasets/dstc8-schema-guided-dialogue`,
  commit `e852981ae34990f4358979625854259302feaa78`
- ATIS: Hugging Face mirror `pfsv/atis`, config `default`, split `test`

完整计数和分域统计见 `summary.json`。筛选代码是
`scripts/extract_external_od_subsets.py`；运行时传入 MultiWOZ 测试文件、
MultiWOZ dialogue acts、SGD test 目录和 ATIS rows API 下载目录即可重建全部文件。
