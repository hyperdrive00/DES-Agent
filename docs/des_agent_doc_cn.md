
# **ChatGPT Agent + Neo4j 使用手册**

## **目录**
1. [概述](#概述)  
2. [功能简介](#功能简介)  
3. [数据库结构（Schema）](#数据库结构schema)  
    - [节点与属性](#节点与属性)  
    - [关系与属性](#关系与属性)  
4. [示例数据](#示例数据)  
5. [典型问答示例](#典型问答示例)  
6. [使用流程](#使用流程)  
    - [主要组件与交互](#主要组件与交互)  


---

## **概述**
本项目通过一个定制的 ChatGPT Agent 来与 Neo4j 数据库交互。用户可以用自然语言提出与数据库中深度共熔溶剂（Deep Eutectic Solvent, DES）及其相关文献、配比、物性信息等有关的问题，Agent 将自动生成并执行相应的 Cypher 查询，并返回结果给用户。

---

## **功能简介**
- **自然语言提问**：无需直接编写 Cypher 语句，Agent 将根据您的问题自动生成查询。  
- **数据库存储与检索**：基于 Neo4j 图数据库，支持对文献信息、配比信息、物质信息等进行结构化存储和查询。  
- **动态问答**：可以在同一上下文中多轮连续提问，获取更加详细或深入的相关信息。  
- **示例查询**：查询 DES 的定义、与哪些物质能形成 DES、某些特定条件下（如熔点范围）的混合物信息等。

---

## **数据库结构（Schema）**
<div align="center">
  <img src="./assets/data_schema.png" alt="数据库结构" />
  <p>数据库结构</p>
</div>


### 节点与属性

本项目在 Neo4j 中使用三个标签（Labels）来表示数据：**Article**、**Mixture** 和 **Substance**。节点与属性结构如下。

#### 1. `Article`
| 属性名         | 说明                                 |
|----------------|--------------------------------------|
| `articleID`    | 文献的唯一 ID                        |
| `articleTitle` | 文献标题                              |
| `articleDOI`   | 文献数字对象标识符（DOI）             |
| `source`       | 文献信息来源（如 “abstract,” “table”） |

#### 2. `Mixture`
| 属性名                   | 说明                                                         |
|--------------------------|--------------------------------------------------------------|
| `mixtureID`             | 混合物在数据库中的唯一 ID                                     |
| `proportionsUnit`       | 比例的单位（如 “%,” “molar ratio”等）                         |
| `meltingPoint`          | 熔点数值（浮点型或字符串）                                    |
| `meltingPointRange`     | 熔点范围（如 “[X, Y]”）                                       |
| `meltingPointUnit`      | 熔点单位（如 “°C,” “K”）                                       |
| `roomTempAspect`        | 室温下的外观或状态（如“液体”，“固体”等）                     |
| `effectiveTemp`         | 有效温度数值（浮点型或字符串）                                |
| `effectiveTempRange`    | 有效温度范围（如 “[X, Y]”）                                   |
| `effectiveTempUnit`     | 有效温度单位（如 “°C,” “K”）                                   |
| `operatingTemperatureRange` | 操作温度范围（可选或遗留字段）                             |
| `waterContent`          | 混合物中含水量的值（字符串或浮点）                             |
| `waterContentUnit`      | 含水量单位（如 “wt%,” “ppm”）                                 |
| `hydrophilicity`        | 对亲水性的描述（如“hydrophilic”，“hydrophobic”）              |
| `avgXlogP`              | 平均 XLogP（浮点型）                                          |
| `avgTPSA`               | 平均 TPSA（浮点型）                                           |

#### 3. `Substance`
| 属性名             | 说明                                                      |
|--------------------|-----------------------------------------------------------|
| `pubChemName`      | 物质在 PubChem 中的名称（唯一）                            |
| `pubChemCID`       | PubChem 化合物编号                                        |
| `MolecularFormula` | 分子式                                                    |
| `MolecularWeight`  | 分子量（浮点数）                                          |
| `IsomericSMILES`   | 物质的同分异构体 SMILES 描述                               |
| `XLogP`            | 油水分配系数（来自 PubChem）                              |
| `TPSA`             | 拓扑极性表面积                                           |
| `Charge`           | 分子净电荷                                               |
| `HBondDonorCount`  | 氢键供体数量                                             |
| `HBondAcceptorCount` | 氢键受体数量                                           |
| `Synonyms`         | 同义词列表                                               |

### 关系与属性

#### 1. `(Mixture)-[:isReportedIn]->(Article)`
- **方向**：从 `Mixture` 指向 `Article`  
- **意义**：该混合物信息来源或被报道于某篇文献。  

#### 2. `(Mixture)-[:hasSubstance { proportion, functionalRole }]->(Substance)`
- **方向**：从 `Mixture` 指向 `Substance`
- **关系属性**：
  - `proportion`：该物质在混合物中的比例值（如 “0.5” 或 “50%”）
  - `functionalRole`：该物质在混合物中的功能角色（如 “HBA” 或 “HBD”）
- **意义**：描述某混合物由哪些物质组成，以及其比例及功能角色。


---

## **示例数据**

以下示例（抽取自真实文献或构造的测试数据）展示了如何组织数据。为了便于阅读，这里提供**表格格式**示例，实际存储时会以节点与关系形式存在于 Neo4j 中。

**Mixture / Article 信息示例：**

| mixture_id | article title                                                                  | article doi               | article id | source   | substance names 1 | substance names 2  | substance names 3 | substance names 4 | proportions 1         | proportions 2         | proportions 3 | proportions 4 | unit of proportions | melting point | melting point (range) | unit of melting point | aspect at room temperature | effective temperature | effective temperature (range) | unit of effective temperature | water content | unit of water content | substance character 1 | substance character 2 | substance character 3 | substance character 4 | xlogp_avg | tpsa_avg | hydrophilicity |
|------------|--------------------------------------------------------------------------------|---------------------------|-----------|----------|-------------------|--------------------|-------------------|-------------------|------------------------|------------------------|---------------|---------------|---------------------|--------------|------------------------|-----------------------|----------------------------|----------------------|----------------------------|-----------------------------|--------------|-----------------------|------------------------|------------------------|------------------------|------------------------|-----------|----------|---------------|
| 0          | Efficient, green extraction of ... with deep eutectic solvents                | 10.1016/j.microc.2022... | 299       | table 1  | Betaine           | 1,2,4-Butanetriol  |                   |                   | 0.3333333333333333     | 0.6666666666666666     |               |               | molar ratio         |              |                        |                       |                            |                      |                            |                             |              |                       | HBA                    |                        |                        |                        | -0.76666  | 53.8333  | hydrophilic   |
| 1          | Efficient, green extraction of ... with deep eutectic solvents                | 10.1016/j.microc.2022... | 299       | table 1  | Choline Chloride  | 1,2,4-Butanetriol  |                   |                   | 0.3333333333333333     | 0.6666666666666666     |               |               | molar ratio         |              |                        |                       |                            |                      |                            |                             |              |                       |                        |                        |                        |                        | -1.4      | 47.2     | hydrophilic   |
| 2          | Efficient, green extraction of ... with deep eutectic solvents                | 10.1016/j.microc.2022... | 299       | table 1  | Proline           | 1,2,4-Butanetriol  |                   |                   | 0.3333333333333333     | 0.6666666666666666     |               |               | molar ratio         |              |                        |                       |                            |                      |                            |                             |              |                       |                        |                        |                        |                        | -1.76666  | 56.9     | hydrophilic   |


**Substance 信息示例：**

| synonyms                           | pubchem name    | pubchem cid | MolecularFormula | MolecularWeight | IsomericSMILES             | XLogP | TPSA    | Charge | HBondDonorCount | HBondAcceptorCount | 
|------------------------------------|-----------------|------------:|------------------|----------------:|----------------------------|-------|---------|--------|-----------------|--------------------|
| ['zinc chloride']                  | Zinc Chloride   | 5727        | Cl2Zn            | 136.3           | Cl[Zn]Cl                   |       | 0.0     | 0      | 0               | 0                  |
| ['diopside']                       | Diopside        | 166740      | CaMgO6Si2        | 216.55          | [O-][Si](=O)[O-]. ...      |       | 126.0   | 0      | 0               | 6                  |
| ['sodium chloride','NaCl']         | Sodium Chloride | 5234        | ClNa            | 58.44           | [Na+].[Cl-]                |       | 0.0     | 0      | 0               | 1                  |

> **提示**：以上表格仅供示例，实际会以 Neo4j 图结构的形式导入，包含节点和关系以及各自的属性。

---

## **典型问答示例**

以下列出一些与 DES（深度共熔溶剂）相关的典型问题示例，用户可以通过 ChatGPT Agent 直接提问，Agent 将自动生成 Cypher 查询并返回答案。

1. **什么是 DES？**  
   - 示例回答：简单解释 DES 定义以及数据库中相关记录。

2. **什么物质可以和尿素组成 DES？**  
   - 示例回答：列出数据库里和“尿素”共同配伍的物质信息。

3. **哪些物质能和氯化胆碱（Choline Chloride）形成 DES？**  
   - 如果需要过滤低熔点条件，可进一步追问：  
     *其中哪些熔点低于 25 度？*  

4. **数据库中熔点最低的 DES 由什么物质组成？配比是多少？**  
   - 示例回答：输出该 DES 的物质名称及其摩尔比（或质量比），并显示相关文献信息。

以上问题示例仅供参考，您可以根据业务需求和数据库内容做更复杂的查询。

---

## **使用流程**

### 主要组件与交互
1. **ChatGPT Agent**：基于 OpenAI/ChatGPT 模型，用于理解自然语言并生成 Cypher 查询。  
2. **Cypher Query Agent**：将 ChatGPT Agent 生成的查询真正发送给 Neo4j 数据库执行，并返回结果。  
3. **Neo4j 数据库**：存储 DES 相关的所有节点、关系及属性。  

<div align="center">
  <img src="./assets/agent_structure.png" alt="Agent的回答流程" />
  <p>Agent的回答流程</p>
</div>
