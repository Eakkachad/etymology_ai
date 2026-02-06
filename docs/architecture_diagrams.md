
# Data Pipeline Architecture

```mermaid
flowchart TB
    subgraph Sources["Data Sources"]
        K[Kaikki<br/>Wiktionary<br/>3.5 GB]
        W[WOLD<br/>Loanwords<br/>5 MB]
        S[Starling<br/>PIE Roots<br/>10 MB]
        P[PanLex<br/>Translations<br/>50 MB]
    end
    
    subgraph Download["Download & Cache"]
        D1[KaikkiDownloader]
        D2[WOLDDownloader]
        D3[StarlingDownloader]
        D4[PanLexDownloader]
    end
    
    subgraph Process["Processing Pipeline"]
        F1[Filter Etymology<br/>Entries]
        F2[Extract<br/>Loanwords]
        F3[Parse PIE<br/>Roots]
        F4[Cache<br/>Queries]
    end
    
    subgraph IPA["IPA Conversion"]
        I1[Thai → IPA]
        I2[Sanskrit → IPA]
        I3[Latin/Greek → IPA]
    end
    
    subgraph Storage["Processed Storage"]
        DB[(PostgreSQL<br/>Etymology DB)]
        PQ[Parquet Files<br/>500 MB]
        GR[Graph Database<br/>NetworkX/Neo4j]
    end
    
    subgraph Training["Model Training Data"]
        CP[Cognate Pairs<br/>Positive + Negative]
        EM[Phonetic<br/>Embeddings]
        GT[Graph<br/>Structure]
    end
    
    K --> D1 --> F1 --> I1 --> PQ
    W --> D2 --> F2 --> DB
    S --> D3 --> F3 --> GR
    P --> D4 --> F4 --> DB
    
    I2 --> PQ
    I3 --> PQ
    
    PQ --> CP
    DB --> CP
    GR --> GT
    PQ --> EM
    
    CP --> Training
    EM --> Training
    GT --> Training
    
    style Sources fill:#e1f5ff
    style Training fill:#ffe1f5
    style Storage fill:#f5ffe1
```

---

# Etymology Chain Structure

```mermaid
graph LR
    subgraph Thai["Thai (Modern)"]
        T1[มารดา<br/>mātṛdā]
        T2[ไตร<br/>trai]
        T3[ทศ<br/>thot]
    end
    
    subgraph Pali["Pali/Sanskrit<br/>(~500 BCE)"]
        S1[mātṛ<br/>मातृ]
        S2[tri<br/>त्रि]
        S3[daśa<br/>दश]
    end
    
    subgraph PIE["Proto-Indo-European<br/>(~4000 BCE)"]
        P1[*méh₂tēr]
        P2[*tréyes]
        P3[*deḱm̥]
    end
    
    subgraph Latin["Latin"]
        L1[māter]
        L2[trēs]
        L3[decem]
    end
    
    subgraph Greek["Ancient Greek"]
        G1[μήτηρ<br/>mētēr]
        G2[τρεῖς<br/>treîs]
        G3[δέκα<br/>déka]
    end
    
    subgraph English["English (Modern)"]
        E1[mother]
        E2[three]
        E3[ten]
    end
    
    T1 -.borrowed.-> S1
    T2 -.borrowed.-> S2
    T3 -.borrowed.-> S3
    
    S1 --> P1
    S2 --> P2
    S3 --> P3
    
    P1 --> L1
    P1 --> G1
    P1 --> E1
    
    P2 --> L2
    P2 --> G2
    P2 --> E2
    
    P3 --> L3
    P3 --> G3
    P3 --> E3
    
    style Thai fill:#ffcccc
    style Pali fill:#ffddaa
    style PIE fill:#ffffcc
    style Latin fill:#ccffcc
    style Greek fill:#ccddff
    style English fill:#ffccff
```

---

# Model Architecture

```mermaid
graph TB
    subgraph Input["Input Layer"]
        W1[Word 1:<br/>มารดา]
        W2[Word 2:<br/>mother]
    end
    
    subgraph Phonetic["Phonetic Conversion"]
        IPA1[IPA:<br/>mɑːdɑː]
        IPA2[IPA:<br/>mʌðər]
    end
    
    subgraph Embedding["Phonetic Embedding<br/>(Transformer)"]
        E1[Encoder 1<br/>6 layers<br/>512-dim]
        E2[Encoder 2<br/>SHARED WEIGHTS<br/>512-dim]
    end
    
    subgraph Vector["Vector Space"]
        V1[φ₁ ∈ ℝ⁵¹²]
        V2[φ₂ ∈ ℝ⁵¹²]
    end
    
    subgraph Similarity["Similarity Scoring"]
        COS[Cosine<br/>Similarity]
        DIST[Distance:<br/>d = 1 - cos(φ₁, φ₂)]
    end
    
    subgraph Output["Cognate Prediction"]
        PROB[P(cognate)]
        CONF[Confidence<br/>Score]
    end
    
    W1 --> IPA1 --> E1 --> V1
    W2 --> IPA2 --> E2 --> V2
    
    V1 --> COS
    V2 --> COS
    COS --> DIST --> PROB --> CONF
    
    subgraph GNN["Graph Neural Network<br/>(Etymology Tree)"]
        N1[Node:<br/>มารดา]
        N2[Node:<br/>mātṛ]
        N3[Node:<br/>*méh₂tēr]
        N4[Node:<br/>mother]
        
        N1 -.edge.-> N2
        N2 -.edge.-> N3
        N3 -.edge.-> N4
    end
    
    CONF -.predict edges.-> GNN
    
    style Input fill:#e1f5ff
    style Embedding fill:#ffe1f5
    style GNN fill:#f5ffe1
    style Output fill:#ffffcc
```
