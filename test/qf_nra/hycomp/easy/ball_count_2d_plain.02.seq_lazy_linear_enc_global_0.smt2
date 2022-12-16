(set-info :smt-lib-version 2.6)
(set-logic QF_NRA)
(set-info :source |Benchmarks generated from hycomp (https://es-static.fbk.eu/tools/hycomp/). BMC instances of non-linear hybrid automata taken from: Alessandro Cimatti, Sergio Mover, Stefano Tonetta, A quantifier-free SMT encoding of non-linear hybrid automata, FMCAD 2012 and Alessandro Cimatti, Sergio Mover, Stefano Tonetta, Quantier-free encoding of invariants for Hybrid Systems, Formal Methods in System Design. This instance solves a BMC problem of depth 0 and uses the quantifier free encoding with equivalences encoding. Contacts: Sergio Mover (mover@fbk.eu), Stefano Tonetta (tonettas@fbk.eu), Alessandro Cimatti (cimatti@fbk.eu).|)
(set-info :category "industrial")
(set-info :status unsat)
;; MathSAT API call trace
;; generated on Mon Mar 19 10:47:20 2012
(declare-fun b.delta__AT0 () Real)
(declare-fun b.EVENT.0__AT0 () Bool)
(declare-fun speed_loss__AT0 () Real)
(declare-fun b.counter.3__AT0 () Bool)
(declare-fun b.speed_y__AT0 () Real)
(declare-fun b.EVENT.1__AT0 () Bool)
(declare-fun b.time__AT0 () Real)
(declare-fun b.y__AT0 () Real)
(declare-fun b.counter.0__AT0 () Bool)
(declare-fun b.counter.1__AT0 () Bool)
(declare-fun g__AT0 () Real)
(declare-fun b.event_is_timed__AT0 () Bool)
(declare-fun b.counter.2__AT0 () Bool)
(assert (let ((.def_63 (* (- 1.0) g__AT0)))
(let ((.def_64 (* (/ 1 2) .def_63)))
(let ((.def_72 (* 2.0 .def_64)))
(let ((.def_73 (* b.delta__AT0 .def_72)))
(let ((.def_74 (+ b.speed_y__AT0 .def_73)))
(let ((.def_78 (<= .def_74 0.0 )))
(let ((.def_77 (<= b.speed_y__AT0 0.0 )))
(let ((.def_79 (and .def_77 .def_78)))
(let ((.def_75 (<= 0.0 .def_74)))
(let ((.def_71 (<= 0.0 b.speed_y__AT0)))
(let ((.def_76 (and .def_71 .def_75)))
(let ((.def_80 (or .def_76 .def_79)))
(let ((.def_66 (* b.speed_y__AT0 b.delta__AT0)))
(let ((.def_61 (* b.delta__AT0 b.delta__AT0)))
(let ((.def_65 (* .def_61 .def_64)))
(let ((.def_67 (+ .def_65 .def_66)))
(let ((.def_68 (+ b.y__AT0 .def_67)))
(let ((.def_69 (<= 0.0 .def_68)))
(let ((.def_55 (<= 0.0 b.y__AT0)))
(let ((.def_70 (and .def_55 .def_69)))
(let ((.def_81 (and .def_70 .def_80)))
(let ((.def_44 (<= g__AT0 10.0 )))
(let ((.def_43 (<= 8.0 g__AT0)))
(let ((.def_45 (and .def_43 .def_44)))
(let ((.def_82 (and .def_45 .def_81)))
(let ((.def_52 (<= speed_loss__AT0 (/ 1 2))))
(let ((.def_49 (<= (/ 1 10) speed_loss__AT0)))
(let ((.def_53 (and .def_49 .def_52)))
(let ((.def_83 (and .def_53 .def_82)))
(let ((.def_54 (and .def_45 .def_53)))
(let ((.def_56 (and .def_54 .def_55)))
(let ((.def_38 (not b.EVENT.0__AT0)))
(let ((.def_36 (not b.EVENT.1__AT0)))
(let ((.def_39 (or .def_36 .def_38)))
(let ((.def_6 (not b.counter.0__AT0)))
(let ((.def_4 (not b.counter.1__AT0)))
(let ((.def_29 (or .def_4 .def_6)))
(let ((.def_33 (or b.counter.3__AT0 .def_29)))
(let ((.def_30 (or b.counter.2__AT0 .def_29)))
(let ((.def_9 (not b.counter.2__AT0)))
(let ((.def_28 (or .def_6 .def_9)))
(let ((.def_31 (and .def_28 .def_30)))
(let ((.def_25 (not b.counter.3__AT0)))
(let ((.def_32 (or .def_25 .def_31)))
(let ((.def_34 (and .def_32 .def_33)))
(let ((.def_40 (and .def_34 .def_39)))
(let ((.def_57 (and .def_40 .def_56)))
(let ((.def_7 (and .def_4 .def_6)))
(let ((.def_10 (and .def_7 .def_9)))
(let ((.def_26 (and .def_10 .def_25)))
(let ((.def_22 (= b.speed_y__AT0 0.0 )))
(let ((.def_19 (= b.y__AT0 10.0 )))
(let ((.def_14 (= b.time__AT0 0.0 )))
(let ((.def_16 (and .def_14 b.event_is_timed__AT0)))
(let ((.def_20 (and .def_16 .def_19)))
(let ((.def_23 (and .def_20 .def_22)))
(let ((.def_27 (and .def_23 .def_26)))
(let ((.def_58 (and .def_27 .def_57)))
(let ((.def_11 (not .def_10)))
(let ((.def_59 (and .def_11 .def_58)))
(let ((.def_84 (and .def_59 .def_83)))
.def_84))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
(check-sat)
(exit)
