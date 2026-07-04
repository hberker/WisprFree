"""Reading prompts for voice-training sessions.

A mix of phonetically rich general sentences (Harvard-sentence style)
and template sentences that embed the user's personal-dictionary terms —
recording yourself saying your own names/jargon is where personal
fine-tuning pays off most.
"""

from __future__ import annotations

import random

BASE_SENTENCES = [
    "The birch canoe slid on the smooth planks.",
    "Glue the sheet to the dark blue background.",
    "It's easy to tell the depth of a well.",
    "These days a chicken leg is a rare dish.",
    "Rice is often served in round bowls.",
    "The juice of lemons makes fine punch.",
    "The box was thrown beside the parked truck.",
    "The hogs were fed chopped corn and garbage.",
    "Four hours of steady work faced us.",
    "A large size in stockings is hard to sell.",
    "The boy was there when the sun rose.",
    "A rod is used to catch pink salmon.",
    "The source of the huge river is the clear spring.",
    "Kick the ball straight and follow through.",
    "Help the woman get back to her feet.",
    "A pot of tea helps to pass the evening.",
    "Smoky fires lack flame and heat.",
    "The soft cushion broke the man's fall.",
    "The salt breeze came across from the sea.",
    "The girl at the booth sold fifty bonds.",
    "The small pup gnawed a hole in the sock.",
    "The fish twisted and turned on the bent hook.",
    "Press the pants and sew a button on the vest.",
    "The swan dive was far short of perfect.",
    "The beauty of the view stunned the young boy.",
    "Two blue fish swam in the tank.",
    "Her purse was full of useless trash.",
    "The colt reared and threw the tall rider.",
    "It snowed, rained, and hailed the same morning.",
    "Read verse out loud for pleasure.",
    "Please schedule the meeting for Wednesday afternoon, not Tuesday.",
    "Can you send me the updated document before the end of the day?",
    "I'll circle back on this next week once we have the numbers.",
    "Let's grab lunch tomorrow if you're free around noon.",
    "The deployment failed twice, so we rolled back to the previous version.",
    "Thanks so much for your help with the quarterly report.",
    "Remind me to call the dentist about rescheduling my appointment.",
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore on sunny days.",
    "We need three hundred and forty-two units by January fifteenth.",
    "My flight departs at seven forty-five in the morning.",
    "The invoice total comes to one thousand two hundred dollars.",
    "Did you get a chance to review the pull request I opened yesterday?",
    "Honestly, I think the second option is better than the first one.",
    "Everyone agreed the proposal needed a stronger executive summary.",
    "The weather forecast calls for thunderstorms late this evening.",
    "Turn left at the second traffic light and continue for two miles.",
    "I'd rather finish this tonight than worry about it over the weekend.",
    "Measure twice and cut once, as the old saying goes.",
    "The committee will announce its final decision on Friday.",
]

DICTIONARY_TEMPLATES = [
    "I mentioned {term} during the standup this morning.",
    "Could you forward the {term} document to the whole team?",
    "We should discuss {term} at tomorrow's meeting.",
    "The latest update to {term} looks really promising.",
    "Ask {term} about the timeline before we commit to anything.",
    "I spent the afternoon working on {term} again.",
    "Everyone keeps asking me questions about {term} lately.",
    "Let's make sure {term} is covered in the release notes.",
]


def generate_prompts(
    dictionary_terms: list[str] | None = None,
    n: int = 20,
    seed: int | None = None,
) -> list[str]:
    """Return ``n`` prompts, interleaving dictionary-term sentences (~1 in 3)."""
    rng = random.Random(seed)
    general = BASE_SENTENCES[:]
    rng.shuffle(general)

    term_sentences: list[str] = []
    if dictionary_terms:
        terms = dictionary_terms[:]
        rng.shuffle(terms)
        templates = DICTIONARY_TEMPLATES[:]
        for i, term in enumerate(terms):
            template = templates[i % len(templates)]
            term_sentences.append(template.format(term=term))

    prompts: list[str] = []
    while len(prompts) < n and (general or term_sentences):
        # every third prompt exercises a dictionary term, when available
        if term_sentences and (len(prompts) % 3 == 2 or not general):
            prompts.append(term_sentences.pop(0))
        elif general:
            prompts.append(general.pop(0))
    return prompts
