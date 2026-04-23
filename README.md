🧩 Koda Puzzle Engine

Enterprise-grade Python generators for 28+ logic, math, and word puzzles. Built to seed the KODA puzzle platform, this engine outputs verified, Supabase-ready .jsonl data at scale.
🌟 Why This Exists

Finding open-source puzzle generators that guarantee exactly one unique solution is incredibly difficult. Most available code generates random grids without verification, leading to ambiguous puzzles. This engine was built from the ground up using constraint satisfaction (Z3-Solver) and optimized backtracking to ensure every single puzzle has a verified, unique solution.
🚀 Features
```
    28+ Games Supported: From Sudoku and Nonograms to Nerdle and Contexto.
    Strict Uniqueness: Every logic puzzle is passed through a solver to guarantee exactly one valid solution.
    Configurable Generation: Generate by size (5x5, 9x9, etc.) and difficulty (Easy, Medium, Hard).
    Database Ready: Outputs directly to .jsonl format matching the Supabase/PostgreSQL schema for easy bulk-insertion.
    Scalable: Designed to pre-generate millions of puzzles efficiently.
```
🛠 Tech Stack
```
    Python 3.9+
    Z3-Solver: For constraint-based logic verification (Sudoku, Hitori, etc.).
    NumPy: For efficient grid and matrix manipulations.
    Gensim / Word2Vec: For semantic distance calculations (Contexto).
```
📦 Installation & Usage

Clone the repository:
```
    git clone https://github.com/yourusername/koda-puzzle-engine.gitcd koda-puzzle-engine
```
Install dependencies: 
 ```bash 
    pip install -r requirements.txt
 ```
  Run a generator: 
  ``` bash  
    python -m generators.sudoku --size 9x9 --difficulty medium --count 1000 --output output/sudoku_medium.jsonl
   ```  
📊 Output Schema 

Every generated line in the .jsonl output adheres to the following schema, ready for database ingestion: 
```json
{
  "game_slug": "sudoku",
  "mode": "infinite",
  "size_key": "9x9",
  "difficulty": "medium",
  "puzzle_data": { "grid": [...] },
  "solution_data": { "grid": [...] }
}
 ``` 
🎮 Supported Games 

Logic & Grid: Sudoku, Hitori, Minesweeper, Nonograms, Light Up, Star Battle, Nurikabe, Pipes, Bridges, Masyu, Skyscrapers, Shikaku, Futoshiki, Fillomino, KenKen, Yajilin, Shakashaka, Dominosa, Tents.
Word & Math: Lexa (Wordle), Nerdle, Quordle, Waffle, Crosswordle, Strands, Crossword, Betweenle, Contexto. 
📄 License 

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 

