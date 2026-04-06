---
title: Surviving Extreme Feature Sparsity and Distribution Shift in CTR Prediction
description: Lessons from a CTR prediction project for a low-activity user segment operating under extreme feature sparsity and distribution shift.
publishDate: 2026-04-05
updatedDate: 2026-04-07
category: Machine Learning
tags:
  - CTR prediction
  - Recommendation systems
  - Data Sparsity
  - Distribution Shift
  - Feature Engineering
featured: true
draft: false
---

Hi, I'm Hangwoo Cho. I work as a Solution Engineer in Microsoft's AI Build team, where I help customers deploy machine learning systems and solve practical ML problems in production.

I recently had a chance to build a CTR prediction model on one of the largest large-scale data flows in Korea, and that setting shaped a lot of what this post is about.

As part of Microsoft's mentoring program, I recently started writing down lessons learned from real machine learning problems I have encountered at work and from the technical discussions around them.

This post is the first note in that series. It is, in a sense, a story about trying to make a CTR model survive inside that dataset rather than just look good on a clean benchmark.

It focuses on two issues that surfaced very quickly in a CTR prediction project:

- **extreme data sparsity**
- **partial distribution shift**

> In real-world CTR systems, the hardest cases are often the ones that matter most: users with the least signal and the greatest mismatch between training and serving conditions.

## Problem context

The project involved a CTR prediction task for a promotional push-notification campaign on a consumer digital platform.

The available data looked reasonable at first glance:

- user behavior features,
- user metadata,
- content metadata,
- user-content interaction features.

But the actual target group was a **low-activity user segment**. In other words, the users we cared about most were the ones with the least recent activity to learn from.

That created the core mismatch immediately: the most important segment was also the one with the weakest behavioral signal.

## What the model actually had to do

Once the business context was clear, the modeling objective itself was straightforward:

- use behavioral and metadata features to predict click-through rate,
- rank users for a campaign,
- and make the system work especially well for low-activity users.

What made this difficult was not the definition of the task, but the characteristics of the target segment. Low-activity users were not just another slice of the same population. They were colder, sparser, and behaviorally different from the general user base.

This was the real heart of the problem: the users who mattered most were exactly the users for whom the model had the least useful signal, and they were also not distributed like the broader population the model would normally learn from.

In practice, that split the modeling challenge into two concrete problems.

**Problem 1 - extreme data sparsity:** the target users had too little in-domain behavioral history for the usual interaction features to be reliable.

**Problem 2 - partial distribution shift:** the users we needed to serve were not distributed like the broader population the model would normally be trained on.


## Problem 1: Extreme data sparsity

> **Problem 1 takeaway:** when in-domain history is too sparse, the most practical path is to recover signal from adjacent domains first, then make sparse identifiers and short behavior sequences more usable through better representations, and only use sampling with calibration in mind.

For the low-activity segment, the most important feature family, user-content interaction history, was also the sparsest.

This is a very common industrial ML pattern. The features that usually drive ranking quality are often the ones that disappear first in cold-start or inactive-user scenarios.

Up to that point, I had mostly approached feature engineering in a Kaggle-like way: work harder on the columns already in front of me. But this case forced a different order of thinking.

The first move was **cross-domain feature discovery**: before trying to squeeze more out of weak target-domain logs, look for usable signal from adjacent domains. After that, the problem became how to make the remaining sparse inputs more usable through better representations such as hashing or embeddings, richer sequence-based features, and, if needed, sampling with calibration in mind.

### Solution 1. Start by looking for cross-domain features

> When the target surface is quiet, listen to the neighboring surfaces.

If the target app or target content does not provide enough behavior, the next best move is to ask: **what other signals still reveal user preference or intent?**

In a real-world setting, unlike many data science competition setups, you are sometimes able to go out and discover additional features. When that option exists, feature discovery often has the biggest impact.

For low-activity users, direct target-app interaction might be weak or nearly missing, but that does not mean the user is completely unobservable. In this project, the more useful signals came from adjacent product surfaces and broader usage context. I cannot disclose the exact feature set, but examples of the patterns that mattered include:

- coarse user profile and account-state signals,
- device and platform usage patterns,
- engagement signals from adjacent app surfaces,
- behavioral traces from related content or campaign touchpoints,
- and short-horizon activity features outside the exact target interaction.

This matters because the sparsity was not absolute. It was **target-domain sparsity**. The user could be nearly blank for the exact notification-click history we wanted to model while still being partially observable through nearby behaviors.

Empirically, this was the highest-impact intervention. Before tuning the model harder, the better move was to expand the observable feature space and recover signal from variables that were correlated with the target task but not limited to the exact serving surface.

### Solution 2. Compress sparse identifiers with better representations

> Better representations can make sparse identifiers more useful, but in a tree-based production path they are often a secondary lever rather than the main one.

The next set of ideas from those discussions was about representation. This was one of the directions I found conceptually appealing, but in this particular project it did not translate into as much practical lift as I initially expected.

I cannot disclose the full model stack in detail, but the production baseline was closer to a distributed LightGBM-style tree system than to a deep representation learner. In that setup, many high-cardinality signals were already entering the model as categorical features, so the marginal benefit of adding another representation layer was naturally more limited.

One direct suggestion was **feature hashing** for very high-cardinality identifiers tied to users, content, sessions, or other interaction entities. Hashing can reduce feature dimensionality and make sparse tabular inputs more manageable.

But there is an important limitation. Hashing mostly compresses one dimension. It does not naturally capture similarity between related users or related items.

That is why the more promising direction was **learned embeddings**.

Instead of feeding raw identifiers directly into the main model, we can train compact representations for things like:

- user IDs,
- session IDs,
- device types,
- history-related entities.

The point is not to over-engineer the system with a giant deep recommender stack. The point is to map sparse, high-cardinality inputs into a denser and more meaningful feature space.

So I still think this was the right direction to consider, but in this case it felt more like a constrained improvement path than the main lever. Compared with cross-domain feature discovery, the gains here were more limited.

### Solution 3. Use a small sequence model as a feature engineering tool

> Sequence models can preserve behavioral structure, but under tight compute and a tree-based production path they made more sense as a feature tool than as the main modeling direction.

Another strong recommendation was to stop collapsing all behavioral logs into overly simple aggregates too early.

I thought this was a reasonable idea, and in a different setting I probably would have tried it. In this project, however, we did not end up pursuing it. The main reason was expected ROI.

At the time, the baseline system was still a tree model and summarized behavior into features like short-window counts or average CTR-like aggregates. That is operationally convenient, but it can throw away sequence structure. Still, adding a separate sequence encoder would have increased both data-preparation complexity and compute cost, and I did not think it would outperform the other, cheaper interventions we already had on the table.

There was also a modeling mismatch. Sequence encoders are most attractive when the downstream model can directly exploit the representation they produce. In our case, the target production path was still tree-based, so the sequence model would have served only as an upstream feature generator rather than as a native part of the learner. That made the projected gain feel more limited.

So the suggestion was not to replace everything with a large end-to-end recommender model. Instead, the advice was more practical:

- build a **small sequence model**,
- encode click history or app usage history with a sliding window,
- optionally include signals such as usage path or time spent,
- and feed the resulting representation back into the existing model as another feature group.

I still think this is a realistic recommendation. It treats sequence modeling as a feature engineering layer rather than a full architectural rewrite. But in this particular case, given the compute budget and the tree-model production path, I viewed it as a lower-ROI option than cross-domain feature discovery or simpler feature-side improvements.

### Solution 4. Use oversampling carefully, and calibrate afterward

> Oversampling can make sparse-label training easier, but if the output still needs probability semantics, calibration becomes part of the solution.

The discussion also touched on **oversampling positive examples** or using bootstrap-style sampling when positives are too rare.

This came with an important caveat. In this project, the client cared not only about ranking quality, but also about whether predicted CTR values could still be interpreted in a probability-like way.

That means oversampling cannot be treated as a free lunch.

The practical interpretation from the discussion was:

- oversampling can make training easier,
- but it changes the effective training distribution,
- so predictions may need to be calibrated back against the original data distribution.

This was a good reminder that there is a difference between **making the learner see enough positive examples** and **preserving the semantics of the final output**.

In our case, this part was ultimately handled later through a calibration step fit on validation data rather than treated as the main lever for improving the model itself.

## Problem 2: Partial distribution shift

This was not just a generic case of training and inference being misaligned.

The awkward part was this:

- the model was trained on the full population,
- the first deployment target was the low-activity segment,
- and the same model might still need to expand back to the broader population later.

So the problem was not simply "train on population A, serve population B".

It was closer to this:

> train on the broad population, serve a narrower shifted subgroup now, but do not completely give up the option of serving the broader population later.

That is why I think **partial distribution shift** is the right description. The serving group was not fully out-of-distribution, but it was different enough that the mismatch mattered.

### Why this was a particularly awkward kind of shift

What made this more than a textbook covariate-shift story was the interaction between three constraints.

- the serving segment was narrower than the full population,
- that segment was also the one with weaker behavioral signal,
- and the production path still cared about eventual reuse outside that segment.

So the problem was not just statistical misalignment. It was also a product and deployment problem.

If the low-activity segment had been the only target forever, the answer would have been easier: just optimize for that subgroup and move on.

If the full population had been the immediate serving target, then training on the full population would have been much less questionable.

The difficulty came from living in between those two situations.

That is the part I think is worth making explicit. The shift was only "partial" in distributional terms, but operationally it created a very real design tension.

### The answer felt obvious, and that was part of the point

In hindsight, the core response sounds almost embarrassingly obvious.

> if the users you are about to serve do not look like the users you trained on, move the training distribution closer to the serving distribution.

That is not a novel modeling insight. I was aware of that while writing this section, and I think it is better to say that directly than to pretend it was some surprising technical breakthrough.

But obvious does not mean unimportant.

In projects like this, the mistake is often not misunderstanding the principle. The mistake is failing to operationalize it early enough.

In practice, this meant treating training-set construction as part of the modeling problem.

The same cross-domain variables that helped recover signal for sparse users also helped define a training cohort that looked more like the low-activity users we actually cared about.

The useful signals included things like:

- demographic variables,
- device signals,
- customer attributes,
- and partial behavior from adjacent surfaces.

They mattered not only as features, but also as a way to pull the learning distribution closer to the serving distribution.

That sounds simple on paper. It still mattered because it changed where effort went.

Instead of treating the full-population dataset as the unquestioned default, it forced a more specific question:

> who should this model really be allowed to learn from if its first job is to serve this subgroup?

### The trade-off we accepted

Once the objective was framed that way, the trade-off became unavoidable.

If we biased the model toward the low-activity segment, there was a real possibility that aggregate performance over the full population would worsen.

I do not think that should be described as a bug. It was the cost of optimizing for the segment that actually mattered in the first deployment.

This was the part that felt more honest to me than elegant.

There was no magical way to make the subgroup-specific objective disappear. We simply accepted that improving relevance for the target segment might come at the expense of a cleaner global metric.

That acceptance also clarified the longer-term design space. If subgroup needs remain materially different, the better answer is often not to keep pretending everything belongs in one shared model.

The cleaner long-run option may be some combination of:

- explicit segment-specific models,
- routing logic across models,
- or at least a shared model whose training distribution is deliberately biased toward the segment that matters most.

So I do not think the main lesson here was a clever solution.

It was recognizing that the problem itself had already constrained the answer. The response was somewhat obvious, and we knew that. What mattered was being explicit about the trade-off and being willing to accept it.

## Final takeaway

If there is one broader lesson I took away from this project, it is this:

real production datasets are much less well-behaved than the datasets that appear in data science competitions or research benchmarks.

The modeling story in this project was still worth writing down. At a high level, it came down to three things:

- identifying what kind of sparsity we were actually dealing with,
- being precise about what kind of distribution shift this was,
- and choosing to accept the trade-off that came with optimizing for the segment that mattered most.

But honestly, modeling was not the most painful part.

The harder part was large-scale data processing under limited compute.

This work ran on Spark, and I hit out-of-memory errors more times than I can count while trying to make the pipeline stable enough to support the modeling work at all. In practice, that engineering struggle felt at least as educational as the model-design questions, and often more painful.

That part probably deserves its own write-up later.

So this post is really only the first half of the story. It is the part about defining the problem correctly and documenting how we approached it.

Finally, I want to thank my mentor, Juwen Wu, and the teammates I cannot name publicly for helping think through the problem and work through the messy parts of the project together.
