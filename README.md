# GLOBALNAV — Environment-Driven Global Multi-Modal Navigation

## 分工

| 组件 | 职责 |
|------|------|
| **LLM** (`gpt-4o-mini`) | 仅理解用户意图：提取起点、终点；模糊时追问 |
| **MobilityEnvironment** | 枚举所有可行交通组合、计算耗时、生成具体路线 |

**交通方式不由 LLM 决定**，由环境根据真实数据生成。

## 数据源

| 数据 | 来源 | 用途 |
|------|------|------|
| 驾车/步行路线 | [OSRM](https://project-osrm.org/) + OpenStreetMap | 具体路段、逐步转向指示 |
| 航班连通性 | [OpenFlights](https://openflights.org/) routes.dat | 验证航线是否存在（非时刻表） |
| 公交/轨道站点 | OSM Overpass API | walk→轻轨→walk 组合 |
| 火车/大巴（估计） | 距离 + 均速启发式 | 无开放 GTFS 时的备选 |

## 交互式 GUI（分段选择）

```bash
conda activate globalnav
cd /Users/zsf/projects/GLOBALNAV
python scripts/run_gui.py --port 8765
# 浏览器打开 http://localhost:8765
```

每段行程独立展示可选交通方式（步行/驾车/公交/火车/组合子路段），鼠标悬停显示耗时，点击切换后**总耗时实时更新**。默认选中 LLM 推荐方案。

## Instruction Follower（VELMA 风格逐步仿真）

长指令按步骤执行：LLM 分解段落目标 → 环境构建路线 → Agent 逐步选择动作（forward/left/takeoff/cruise…）。

**Web GUI：** `http://localhost:8765/follower`
- Leaflet 地图轨迹（步行 OSRM / 飞行大圆 / 火车估计线）
- 分段交通选项切换（seg_access / seg_flight / seg_egress）
- 单步 / 播放 / 时间轴回放
- **全球街景**（非 VELMA 曼哈顿限定）：Google Street View Metadata + Static API；可选 Mapillary 备选

```bash
# 命令行（规则策略，不消耗 action API）
python scripts/run_follower.py \
  -i "Walk from UTS to Sydney Opera House" \
  --rule-based --offline -v

# UTS → 利物浦 完整多模态（LLM decompose + 网络）
python scripts/run_follower.py \
  -i "我在悉尼科技大学UTS，面朝乔治街。先走到悉尼机场，坐飞机到曼彻斯特，出机场后坐火车到利物浦，最后走到利物浦大学。" \
  --rule-based -v -o output/liverpool_trace.json
```

街景 API（任选其一，写入 `VELMA/.env`）：
- `GOOGLE_MAPS_API_KEY` — Street View Metadata + Static（推荐）
- `MAPILLARY_ACCESS_TOKEN` — 无 GSV 区域备选

| 模块 | 文件 | 职责 |
|------|------|------|
| Decomposer | `globe_nav/follower/decomposer.py` | LLM 将长指令拆为 segment goals + 朝向/转向备注 |
| Alignment | `globe_nav/follower/alignment.py` | 分解目标对齐 planner 三段 ID |
| StreetView | `globe_nav/maps/streetview.py` | lat/lon+heading → 全球街景（GSV / Mapillary） |
| Simulator | `globe_nav/follower/simulator.py` | 沿 OSRM geometry 逐步前进；飞行/火车为离散 phase |
| Verbalizer | `globe_nav/follower/verbalizer.py` | VELMA 风格 observation 文本 |
| Agent | `globe_nav/follower/agent.py` | 观察 → LLM/规则动作 → `env.step()` 循环 |

## GlobNav-Bench 数据收集

```bash
# 生成 500 条 pilot benchmark 数据 + schema + validation report
python scripts/generate_globnav_bench.py --count 500 --seed 7

# 启动标注页面
python scripts/run_gui.py --port 8765
# 浏览器打开 http://localhost:8765/bench
```

输出文件：
- `data/globnav_bench/pilot_500.jsonl` — 全球 route skeleton + instruction + labels
- `data/globnav_bench/schema.json` — JSONL schema
- `data/globnav_bench/pilot_500_validation_report.json` — 覆盖率和校验报告
- `data/globnav_bench/annotations.review.jsonl` — 标注页面保存的人工复核结果

## 命令行（分段规划）

```bash
# 多模态：环境列出所有完整方案（含具体路线）
python scripts/run_navigation.py -i "我在UTS，要去四川九寨沟" -v

# 短途：多种本地出行组合
python scripts/run_navigation.py -i "从UTS到悉尼歌剧院" \
  --origin "University of Technology Sydney" \
  --destination "Sydney Opera House" -v
```

API key 从 `VELMA/.env` 自动加载。

## 部署到 Render

仓库包含 `render.yaml`，可在 Render 里选择 **New Blueprint** 并连接这个 GitHub 仓库部署。

必填环境变量：
- `OPENAI_API_KEY` — LLM 用于解析用户出发地、目的地和澄清问题。

可选环境变量：
- `GOOGLE_MAPS_API_KEY` — Street View Metadata + Static API。
- `MAPILLARY_ACCESS_TOKEN` — Mapillary 街景备选。

Render 启动命令：

```bash
gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 120
```

## 输出示例

```
--- option_1: drive + fly + fly + train (1-stop: SYD→JZH) ---
    Total: 14.0 h | 9008 km
    [1] [drive] UTS → Sydney Airport (12.6 km) [✓ OSM/OSRM]
          → depart on King Street
          → turn on Hunter Street
          ...
    [2] [fly] SYD → CKG [✓ OpenFlights (connectivity)]
          Note: SYD-CKG route on record; 90 min layover
    [3] [fly] CKG → JZH [✓ OpenFlights]
    [4] [train] JZH Airport → Jiuzhaigou (~63 km) [~ estimated]
```

要点：
- **SYD→成都无直飞**：环境查 OpenFlights，自动给出 1-stop 方案（如 SYD→CKG→JZH）
- **每段有多种选择**：去机场可 walk / drive；落地后可 train / drive / bus
- **具体路线**：OSRM 返回真实 OSM 路网转向

## 局限

- 航班：仅有航线是否存在，无实时航班/票价（可用 Amadeus API 扩展）
- 火车/公交：无中国区开放 GTFS 时为时间估计
- OSRM 公共服务器有速率限制，结果会缓存到 `data/cache/`
- Follower 街景：依赖 Google/Mapillary 覆盖；飞行段为 phase 仿真 + 大圆轨迹，机舱内无实景
- VELMA 原版 CLIP 地标 verbalization 仅适用于曼哈顿 Touchdown 图；GLOBALNAV 用 GSV 图像 + OSM 文本
