abstract type AbstractXAIMetric end

# Abstract Types for Quantus Categories
# (a) Faithfulness (b) Robustness (c) Localisation (d) Complexity (e) Randomisation (f) Axiomatic

abstract type AbstractFaithfulnessMetric <: AbstractXAIMetric end
abstract type AbstractRobustnessMetric <: AbstractXAIMetric end
abstract type AbstractLocalisationMetric <: AbstractXAIMetric end
abstract type AbstractComplexityMetric <: AbstractXAIMetric end
abstract type AbstractRandomisationMetric <: AbstractXAIMetric end
abstract type AbstractAxiomaticMetric <: AbstractXAIMetric end
