
# **ChatGPT Agent + Neo4j User Manual**

## **Table of Contents**
- [**ChatGPT Agent + Neo4j User Manual**](#chatgpt-agent--neo4j-user-manual)
  - [**Table of Contents**](#table-of-contents)
  - [**Overview**](#overview)
  - [**Feature Introduction**](#feature-introduction)
  - [**Database Schema**](#database-schema)
    - [Nodes and Properties](#nodes-and-properties)
      - [1. `Article`](#1-article)
      - [2. `Mixture`](#2-mixture)
      - [3. `Substance`](#3-substance)
    - [Relationships and Their Attributes](#relationships-and-their-attributes)
      - [1. `(Mixture)-[:isReportedIn]->(Article)`](#1-mixture-isreportedin-article)
      - [2. `(Mixture)-[:hasSubstance { proportion, functionalRole }]->(Substance)`](#2-mixture-hassubstance--proportion-functionalrole--substance)
  - [**Example Data**](#example-data)
  - [**Typical Q\&A Examples**](#typical-qa-examples)
  - [**Usage Workflow**](#usage-workflow)
    - [Key Components and Interactions](#key-components-and-interactions)

---

## **Overview**
This project implements a customized ChatGPT Agent to interact with a Neo4j database. Users can ask questions in natural language regarding deep eutectic solvents (DES) and their related literature, formulations, and physicochemical information. The Agent will automatically generate and execute the corresponding Cypher queries and return the results to the user.

---

## **Feature Introduction**
- **Natural Language Querying**: No need to write Cypher queries directly; the Agent automatically generates the query based on your question.
- **Database Storage and Retrieval**: Based on the Neo4j graph database, it supports structured storage and querying of literature information, formulation data, substance data, etc.
- **Dynamic Q&A**: It supports multi-turn interactions within the same context to provide more detailed or deeper related information.
- **Example Queries**: Query definitions of DES, identify substances that can form DES with specific compounds, or filter for mixtures under certain conditions (e.g., within a melting point range).

---

## **Database Schema**
<div align="center">
  <img src="./assets/data_schema.png" alt="Database Schema" />
  <p>Database Schema</p>
</div>

### Nodes and Properties

This project uses three labels in Neo4j to represent data: **Article**, **Mixture**, and **Substance**. Their respective node structures and properties are as follows.

#### 1. `Article`
| Property Name    | Description                                  |
|------------------|----------------------------------------------|
| `articleID`      | Unique ID for the article                    |
| `articleTitle`   | Title of the article                         |
| `articleDOI`     | Digital Object Identifier (DOI) of the article|
| `source`         | Source of the literature (e.g., “abstract,” “table”) |

#### 2. `Mixture`
| Property Name                   | Description                                                    |
|---------------------------------|----------------------------------------------------------------|
| `mixtureID`                     | Unique ID of the mixture in the database                       |
| `proportionsUnit`               | Unit for the proportions (e.g., “%,” “molar ratio,” etc.)         |
| `meltingPoint`                  | Melting point value (either as float or string)                   |
| `meltingPointRange`             | Melting point range (e.g., “[X, Y]”)                             |
| `meltingPointUnit`              | Unit for the melting point (e.g., “°C,” “K”)                     |
| `roomTempAspect`                | Appearance or state at room temperature (e.g., “liquid,” “solid”)  |
| `effectiveTemp`                 | Effective temperature value (either as float or string)            |
| `effectiveTempRange`            | Effective temperature range (e.g., “[X, Y]”)                         |
| `effectiveTempUnit`             | Unit for the effective temperature (e.g., “°C,” “K”)                |
| `operatingTemperatureRange`     | Operating temperature range (optional or legacy field)             |
| `waterContent`                  | Water content value in the mixture (string or float)                |
| `waterContentUnit`              | Unit for water content (e.g., “wt%, ” “ppm”)                      |
| `hydrophilicity`                | Description of hydrophilicity (e.g., “hydrophilic,” “hydrophobic”) |
| `avgXlogP`                      | Average XLogP (float)                                             |
| `avgTPSA`                       | Average TPSA (float)                                              |

#### 3. `Substance`
| Property Name       | Description                                         |
|---------------------|-----------------------------------------------------|
| `pubChemName`       | Name of the substance in PubChem (unique)           |
| `pubChemCID`        | PubChem Compound Identifier                         |
| `MolecularFormula`  | Molecular formula                                   |
| `MolecularWeight`   | Molecular weight (float)                            |
| `IsomericSMILES`    | Isomeric SMILES string description of the substance |
| `XLogP`             | Partition coefficient (from PubChem)                |
| `TPSA`              | Topological polar surface area                     |
| `Charge`            | Net molecular charge                                |
| `HBondDonorCount`   | Count of hydrogen bond donors                       |
| `HBondAcceptorCount`| Count of hydrogen bond acceptors                    |
| `Synonyms`          | List of synonyms                                    |

### Relationships and Their Attributes

#### 1. `(Mixture)-[:isReportedIn]->(Article)`
- **Direction**: From `Mixture` to `Article`  
- **Meaning**: Indicates the source document in which the mixture information is reported.

#### 2. `(Mixture)-[:hasSubstance { proportion, functionalRole }]->(Substance)`
- **Direction**: From `Mixture` to `Substance`
- **Relationship Attributes**:
  - `proportion`: The proportion value of this substance in the mixture (e.g., “0.5” or “50%”)
  - `functionalRole`: The functional role of this substance in the mixture (e.g., “HBA” or “HBD”)
- **Meaning**: Describes which substances make up a given mixture along with their proportions and roles.

---

## **Example Data**

Below are examples (extracted from real literature or constructed test data) that show how the data is organized. For ease of reading, the examples are provided in a **table format**. In the actual database, the data is stored as nodes and relationships in Neo4j.

**Mixture / Article Data Example:**

| mixture_id | article title                                                                 | article doi               | article id | source   | substance names 1 | substance names 2  | substance names 3 | substance names 4 | proportions 1         | proportions 2         | proportions 3 | proportions 4 | unit of proportions | melting point | melting point (range) | unit of melting point | aspect at room temperature | effective temperature | effective temperature (range) | unit of effective temperature | water content | unit of water content | substance character 1 | substance character 2 | substance character 3 | substance character 4 | xlogp_avg | tpsa_avg | hydrophilicity |
|------------|-------------------------------------------------------------------------------|---------------------------|-----------|----------|-------------------|--------------------|-------------------|-------------------|------------------------|------------------------|---------------|---------------|---------------------|--------------|------------------------|-----------------------|----------------------------|----------------------|----------------------------|-----------------------------|--------------|-----------------------|------------------------|------------------------|------------------------|------------------------|-----------|----------|---------------|
| 0          | Efficient, green extraction of ... with deep eutectic solvents                | 10.1016/j.microc.2022...  | 299       | table 1  | Betaine           | 1,2,4-Butanetriol  |                   |                   | 0.3333333333333333     | 0.6666666666666666     |               |               | molar ratio         |              |                        |                       |                            |                      |                            |                             |              |                       | HBA                    |                        |                        |                        | -0.76666  | 53.8333  | hydrophilic   |
| 1          | Efficient, green extraction of ... with deep eutectic solvents                | 10.1016/j.microc.2022...  | 299       | table 1  | Choline Chloride  | 1,2,4-Butanetriol  |                   |                   | 0.3333333333333333     | 0.6666666666666666     |               |               | molar ratio         |              |                        |                       |                            |                      |                            |                             |              |                       |                        |                        |                        |                        | -1.4      | 47.2     | hydrophilic   |
| 2          | Efficient, green extraction of ... with deep eutectic solvents                | 10.1016/j.microc.2022...  | 299       | table 1  | Proline           | 1,2,4-Butanetriol  |                   |                   | 0.3333333333333333     | 0.6666666666666666     |               |               | molar ratio         |              |                        |                       |                            |                      |                            |                             |              |                       |                        |                        |                        |                        | -1.76666  | 56.9     | hydrophilic   |

**Substance Data Example:**

| synonyms                           | pubchem name    | pubchem cid | MolecularFormula | MolecularWeight | IsomericSMILES             | XLogP | TPSA    | Charge | HBondDonorCount | HBondAcceptorCount |
|------------------------------------|-----------------|------------:|------------------|----------------:|----------------------------|-------|---------|--------|-----------------|--------------------|
| ['zinc chloride']                  | Zinc Chloride   | 5727        | Cl2Zn            | 136.3           | Cl[Zn]Cl                   |       | 0.0     | 0      | 0               | 0                  |
| ['diopside']                       | Diopside        | 166740      | CaMgO6Si2        | 216.55          | [O-][Si](=O)[O-]…          |       | 126.0   | 0      | 0               | 6                  |
| ['sodium chloride','NaCl']         | Sodium Chloride | 5234        | ClNa             | 58.44           | [Na+].[Cl-]                |       | 0.0     | 0      | 0               | 1                  |

> **Note**: The above tables are provided as examples. In the actual import into Neo4j, the data will be structured as nodes and relationships with their respective properties.

---

## **Typical Q&A Examples**

Below are some typical questions related to DES (deep eutectic solvents) that users might ask using the ChatGPT Agent. The Agent will automatically generate the corresponding Cypher queries and return the answers.

1. **What is DES?**  
   - Example Answer: A brief explanation of the DES definition as well as the related records in the database.

2. **Which substances can form a DES together with urea?**  
   - Example Answer: Lists the substances in the database that are used in formulations with “urea.”

3. **Which substances can form DES with Choline Chloride?**  
   - If further filtering for a low melting point is required, you might ask:  
     *Which of these have a melting point below 25°C?*

4. **What is the DES with the lowest melting point in the database and what are its components and ratios?**  
   - Example Answer: Outputs the names of the substances in the DES along with their molar ratios (or mass ratios) and displays the relevant article information.

The above examples are for reference only; you can generate more complex queries based on your business needs and database content.

---

## **Usage Workflow**

### Key Components and Interactions
1. **ChatGPT Agent**: Based on the OpenAI/ChatGPT model, it is used to understand natural language and generate Cypher queries.
2. **Cypher Query Agent**: Executes the Cypher queries generated by the ChatGPT Agent by sending them to the Neo4j database and returns the results.
3. **Neo4j Database**: Stores all the nodes, relationships, and properties related to DES.

<div align="center">
  <img src="./assets/agent_structure.png" alt="Agent Interaction Flow" />
  <p>Agent Interaction Flow</p>
</div>