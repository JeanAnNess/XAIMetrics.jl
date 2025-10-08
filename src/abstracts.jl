abstract type AbstractXAIEvaluationMetric end

# Abstract Types for Quantus Categories
# (a) Faithfulness (b) Robustness (c) Localisation (d) Complexity (e) Randomisation (f) Axiomatic

abstract type AbstractFaithfulnessMetric <: AbstractXAIEvaluationMetric end
abstract type AbstractRobustnessMetric <: AbstractXAIEvaluationMetric end
abstract type AbstractLocalisationMetric <: AbstractXAIEvaluationMetric end
abstract type AbstractComplexityMetric <: AbstractXAIEvaluationMetric end
abstract type AbstractRandomisationMetric <: AbstractXAIEvaluationMetric end
abstract type AbstractAxiomaticMetric <: AbstractXAIEvaluationMetric end