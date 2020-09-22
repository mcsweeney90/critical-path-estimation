(TeX-add-style-hook
 "critical-path-estimates"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "12pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("hyperref" "colorlinks" "urlcolor=blue" "linkcolor=blue" "citecolor=hotpink") ("babel" "american") ("algorithm2e" "linesnumbered" "ruled") ("tcolorbox" "most" "minted")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art12"
    "a4"
    "amsmath"
    "amssymb"
    "xcolor"
    "graphicx"
    "hyperref"
    "booktabs"
    "rotating"
    "caption"
    "babel"
    "algorithm2e"
    "epstopdf"
    "mathtools"
    "subfig"
    "tcolorbox"
    "mdwlist")
   (TeX-add-symbols
    "R"
    "C"
    "P"
    "E"
    "nbyn"
    "mbyn"
    "l"
    "norm"
    "normi"
    "normo"
    "Chat"
    "e"
    "diag"
    "trace"
    "At"
    "normt"
    "qedsymbol"
    "oldtabcr"
    "nonumberbreak"
    "mynewline"
    "lineref"
    "myvspace"
    "colon")
   (LaTeX-add-labels
    "sect.intro"
    "plot.simple_example"
    "plot.example_fixed"
    "plot.heft_schedule_example"
    "sect.optimistic"
    "eq.opt_uia"
    "eq.opt_ui"
    "tb.opt_example"
    "plot.heft_schedule_example_LB"
    "sect.alt_rankings"
    "eq.expected_node"
    "eq.expected_edge"
    "eq.ur_expectation"
    "subsect.sharper_bounds"
    "subsubsect.monte_carlo"
    "subsubsect.fulkerson"
    "plot.example_edge_only"
    "eq.ur_edge_only"
    "eq.ei"
    "eq.f_fulkerson"
    "prop.fulkerson"
    "para.fulkerson_example"
    "tb.fulk_example"
    "para.fulkerson_computing"
    "eq.f_clingen"
    "para.fulkerson_extensions"
    "alg.fulkerson"
    "subsect.adjusting"
    "eq.expected_node_wm"
    "eq.expected_edge_wm"
    "tb.weighted_example"
    "sect.processor_selection"
    "eq.peft_lookahead"
    "eq.peft_ranks"
    "subsect.alt_cond_cp"
    "eq.cia_min"
    "eq.cia_mean"
    "eq.cia_weighted"
    "eq.s_hat"
    "subsect.ps_priorities"
    "eq.alt_prios"
    "sect.results"
    "subsect.graphs"
    "subsect.benchmarking"
    "plot.bench_slr"
    "plot.bench_speedup"
    "plot.bench_slr_speedup"
    "tb.bench_failures"
    "subsect.evaluation"
    "subsubsect.small_scale"
    "plot.apd_by_ccr"
    "tb.small_rankings_bests"
    "subsubsect.full_set"
    "plot.apd_by_q"
    "subsect.results_selection"
    "subsect.conclusions")
   (LaTeX-add-environments
    "proof"
    "lemma"
    "theorem"
    "prop"
    "code")
   (LaTeX-add-bibliographies
    "references"
    "strings")
   (LaTeX-add-counters
    "mylineno")
   (LaTeX-add-xcolor-definecolors
    "hotpink")
   (LaTeX-add-tcbuselibraries
    "listings"))
 :latex)

