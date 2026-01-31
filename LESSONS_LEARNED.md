# Lessons Learned: Atlas-MAG Implementation

**Project**: Atlas-MAG with Omega Rule (arXiv:2505.23735)
**Date**: January 2026
**Purpose**: Workflow lessons for future paper-to-code implementations (especially HOPE)

---

## Executive Summary

This project successfully implemented a paper-faithful Atlas-MAG model, but accumulated significant "context rot" along the way - ~44% of the codebase became dead code, misleading documentation, or stale tooling. This document captures workflow lessons to avoid these problems in future implementations.

**Key Metric**: Started with ~13,500 lines, cleaned to ~7,600 lines. Almost half was waste.

---

## The Core Problem: Context Rot

Context rot occurs when:
1. **Dead code** accumulates from abandoned experiments
2. **Documentation** is written based on assumptions, not validated understanding
3. **Tooling** is built for architectures that later change
4. **Experiments** that don't work out stay in the codebase "just in case"

**Why it hurts AI-assisted development specifically:**
- Every file read consumes context window
- Misleading docs actively confuse reasoning
- Stale code creates false dependencies
- The AI considers dead code paths when making decisions

---

## Lesson 1: Understand the Paper BEFORE Writing Docs

### What Went Wrong
We wrote extensive documentation (`docs/MAG_MAL_ARCHITECTURE.md`, etc.) based on a misreading of the paper. The Atlas paper describes MAG and MAL as **alternative architectures**, not layers to be interleaved. We implemented interleaving, wrote docs explaining interleaving, then spent significant effort on something the paper never suggested.

### The Pattern
```
Misread paper → Write docs → Docs become "truth" → Build on false foundation
```

### The Fix
1. **No architecture docs until code is working and validated**
2. Use the paper + code docstrings as the single source of truth
3. When in doubt, query the paper directly (HADES semantic search)
4. External docs rot - inline docstrings with paper references don't

### Practical Rule
```
If you can't point to a specific equation/section in the paper, don't implement it.
```

---

## Lesson 2: Identify Architectural Primitives FIRST

### What Went Wrong
We started coding before clearly identifying what the irreducible primitives were at each abstraction level. This led to:
- `MAGBlock` and `MALBlock` as separate classes when only MAG was needed
- Multiple trainer variants that did slightly different things
- Analysis tools built for the wrong architecture

### The Primitive Development Approach
Before writing ANY code, answer:

1. **What are the primitives at each level?**
   ```
   Level 0: Core math operations (outer products, Newton-Schulz)
   Level 1: Memory modules (AtlasMemoryPoly, QKProjection)
   Level 2: Block types (MAGBlock only - not MAL)
   Level 3: Model assembly (skeleton)
   Level 4: Training loop
   ```

2. **What does the paper ACTUALLY specify vs. what's our choice?**
   ```
   Paper specifies: Omega Rule equation, polynomial features, Muon optimizer
   Our choice: SwiGLU for FFN (paper doesn't specify), decay_base value
   ```

3. **What would require bypassing these primitives?**
   If an implementation requires bypassing a primitive, the primitive is wrong.

### Practical Rule
```
Spend 20% of project time on primitive identification before writing code.
```

---

## Lesson 3: Worktrees for Experiments, Not Branches

### What Went Wrong
Experimental code (polarization loss, MAL blocks, various trainers) was developed on branches and merged to main. When experiments failed, the code stayed because "it might be useful later."

### The Fix: Worktree Discipline
```
main              → Only working, validated code. Never experimental.
worktree/exp-xyz  → Experimental. Throwaway BY DEFAULT.
```

**Rules:**
1. Create a worktree for any experiment that might not work
2. If the experiment fails, `git worktree remove --force` - don't merge
3. If it succeeds, rewrite clean on main (don't merge experimental code)
4. Worktrees are cheap. Use them liberally.

### Practical Rule
```
If you're not sure it will work, it goes in a worktree that you're prepared to delete.
```

---

## Lesson 4: Rewrite, Don't Patch

### What Went Wrong
When `MAGBlock` had issues, we patched it repeatedly:
- Added conditional logic
- Added compatibility shims
- Added "temporary" fixes

Each patch made the code harder to understand and introduced more potential for bugs.

### The Rewrite Rule
**When a class needs more than 2-3 patches to fix, delete the file and rewrite it.**

The time spent:
- Understanding why patches aren't working
- Debugging interactions between patches
- Maintaining patch compatibility

...exceeds the time to rewrite from clean understanding.

### The Decision Framework
```
Patch when: Single bug, clear fix, doesn't change architecture
Rewrite when: Multiple issues, unclear root cause, or architecture was wrong
```

### Practical Rule
```
If you're adding a third patch to the same class, stop and rewrite instead.
```

---

## Lesson 5: Kill Experiments Immediately

### What Went Wrong
Polarization loss was an experiment. It didn't work as expected. But it stayed in the codebase for weeks because:
- "We might need it later"
- "It's not hurting anything"
- "We already wrote tests for it"

It WAS hurting - every session spent context on it, considered it in decisions, and maintained compatibility with it.

### The Fix
**If an experiment doesn't work, remove it in the same session.**

Not "comment it out." Not "move it to a utils file." DELETE IT.

If you need it later, you can:
1. Find it in git history
2. Find it in Acheron (archive)
3. Rewrite it (probably better the second time)

### Practical Rule
```
Dead code in the repo is WORSE than no code. Remove it immediately.
```

---

## Lesson 6: Archive Aggressively (Acheron Pattern)

### What Worked
Having a dedicated archive directory (`Acheron/`) made it psychologically easier to delete code. The code wasn't "gone" - it was archived.

### The Archive Pattern
```
~/olympus/
├── Atlas-MAG_OmegaRule/     # Active project - clean
└── Acheron/
    └── Atlas-MAG_OmegaRule_pre-working_clean/  # Archived dead code
        ├── docs_misread_archive/
        ├── analysis_stale/
        └── ARCHIVE_README.md
```

### Rules for Archiving
1. Archive BEFORE deleting (psychological safety)
2. Write a brief README explaining why it was archived
3. Never copy FROM archive to active code (rewrite instead)
4. Archive is write-once, read-never (in practice)

### Practical Rule
```
When in doubt, archive it. Then delete from main. You'll never look at the archive.
```

---

## Lesson 7: Single Source of Truth

### What Went Wrong
We had multiple sources of "truth":
- Paper (actual truth)
- `docs/` directory (based on misreading)
- Code comments (sometimes outdated)
- README.md (lagged behind code)

When sources conflicted, confusion resulted.

### The Fix: Paper + Code Docstrings Only
```python
"""
Omega Rule Q-K Memory Projection.

Paper Reference:
    Atlas: Learning to Optimally Memorize the Context at Test Time
    arXiv:2505.23735, Section 3.2 "Omega Rule", Equation 9

Mathematical Formulation:
    M_t = M_persistent + Σ(i=t-c+1 to t) γ^(t-i) * (k_i ⊗ k_i)

Implementation Notes:
    This is the linear memory special case of Eq. 9...
"""
```

**Every module docstring should have:**
1. Paper reference (arXiv ID, section, equation)
2. Mathematical formulation (what the paper says)
3. Implementation notes (how we implement it, any deviations)

### Practical Rule
```
If it's not in the paper or the code docstrings, it doesn't exist.
```

---

## Lesson 8: Validate Understanding with Semantic Search

### What Worked
Using HADES to query the paper directly caught the MAG/MAL misunderstanding:

```bash
hades db query "MAG MAL interleave alternate layer" --paper 2505.23735 --hybrid
```

Paper says: "two hybrid variants of MAL and MAG" (alternatives)
We read: "interleave MAL and MAG layers" (wrong)

### The Validation Pattern
Before implementing any architectural decision:
1. Form a hypothesis about what the paper says
2. Query the paper with HADES
3. Read the actual text, not your interpretation
4. If the paper doesn't say it, don't implement it

### Practical Rule
```
Trust the paper text over your memory of the paper.
```

---

## Recommended Workflow for Future Projects (HOPE)

### Phase 0: Paper Study (No Code)
1. Ingest paper into HADES
2. Identify all equations that need implementation
3. Create a mapping: equation → code module
4. Identify what the paper specifies vs. what's implementer choice
5. **Query the paper for any assumptions before accepting them**

### Phase 1: Primitive Identification (No Code)
1. List primitives at each abstraction level
2. Define interfaces between levels
3. Identify what would require bypassing primitives (design smell)
4. **Get this reviewed before writing any code**

### Phase 2: Core Implementation (Worktrees)
1. Each major component gets its own worktree
2. Implement against paper equations, not documentation
3. Test against paper examples if available
4. **If it doesn't work, delete the worktree - don't merge broken code**

### Phase 3: Integration (Main Branch)
1. Only merge validated, working components
2. Rewrite clean on main (don't merge experimental code directly)
3. Docstrings reference paper sections/equations
4. **No external docs - paper + code docstrings only**

### Phase 4: Validation
1. Compare outputs to paper results (if available)
2. Run ablations the paper ran
3. **If something doesn't match, question the implementation, not the paper**

---

## Summary: The Three Rules

1. **Understand before implementing**: No code until primitives are identified and paper is understood.

2. **Delete aggressively**: Dead code, failed experiments, and misleading docs hurt more than help. Archive and remove.

3. **Rewrite over patch**: If something needs significant fixing, delete and rewrite. The time investment is lower and the result is better.

---

## Metrics to Track

For future projects, track:
- Lines of code added vs. removed over time
- Number of worktrees created vs. deleted vs. merged
- Time spent on paper study vs. coding
- Number of times a file was "patched" vs. "rewritten"

**Target**: Remove more code than you add in cleanup phases. If cleanup only removes 10% of code, you're not being aggressive enough.

---

## References

- Atlas Paper: arXiv:2505.23735
- Titans Paper: arXiv:2501.00663
- HOPE Paper (next project): arXiv:2512.24695
- Acheron Archive: `~/olympus/Acheron/Atlas-MAG_OmegaRule_pre-working_clean/`
