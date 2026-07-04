"""Reading prompts for voice-training sessions.

Three sources, best-effort in this order:
  1. LLM-generated sentences (local Ollama) that weave in the user's
     personal-dictionary terms — unlimited variety;
  2. a large curated corpus of phonetically balanced (Harvard-style) and
     everyday dictation sentences — the offline fallback;
  3. template sentences embedding dictionary terms.

Recording yourself saying your own names/jargon is where personal
fine-tuning pays off most, so dictionary terms are always represented.
"""

from __future__ import annotations

import logging
import random
import re

log = logging.getLogger(__name__)

BASE_SENTENCES = [
    # Harvard / IEEE phonetically balanced sentences
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
    "Hoist the load to your left shoulder.",
    "Take the winding path to reach the lake.",
    "Note closely the size of the gas tank.",
    "Wipe the grease off his dirty face.",
    "Mend the coat before you go out.",
    "The wrist was badly strained and hung limp.",
    "The stray cat gave birth to kittens.",
    "The young girl gave no clear response.",
    "The meal was cooked before the bell rang.",
    "What joy there is in living.",
    "A king ruled the state in the early days.",
    "The ship was torn apart on the sharp reef.",
    "Sickness kept him home the third week.",
    "The wide road shimmered in the hot sun.",
    "The lazy cow lay in the cool grass.",
    "Lift the square stone over the fence.",
    "The rope will bind the seven books at once.",
    "Hop over the fence and plunge in.",
    "The friendly gang left the drug store.",
    "Mesh wire keeps chicks inside.",
    "The frosty air passed through the coat.",
    "The crooked maze failed to fool the mouse.",
    "Adding fast leads to wrong sums.",
    "The show was a flop from the very start.",
    "A saw is a tool used for making boards.",
    "The wagon moved on well-oiled wheels.",
    "March the soldiers past the next hill.",
    "A cup of sugar makes sweet fudge.",
    "Place a rosebush near the porch steps.",
    "Both lost their lives in the raging storm.",
    "We talked of the side show in the circus.",
    "Use a pencil to write the first draft.",
    "He ran half way to the hardware store.",
    "The clock struck to mark the third period.",
    "A small creek cut across the field.",
    "Cars and busses stalled in snow drifts.",
    "The set of china hit the floor with a crash.",
    "This is a grand season for hikes on the road.",
    "The dune rose from the edge of the water.",
    "Those words were the cue for the actor to leave.",
    "A yacht slid around the point into the bay.",
    "The two met while playing on the sand.",
    "The ink stain dried on the finished page.",
    "The walled town was seized without a fight.",
    "The lease ran out in sixteen weeks.",
    "A tame squirrel makes a nice pet.",
    "The horse trotted around the field at a brisk pace.",
    "Find the twin who stole the pearl necklace.",
    "Cut the cord that binds the box tightly.",
    "The red tape bound the smuggled food.",
    "Look in the corner to find the tan shirt.",
    "The cold drizzle will halt the bond drive.",
    "Nine men were hired to dig the ruins.",
    "The junk yard had a mouldy smell.",
    "The flint sputtered and lit a pine torch.",
    "Soak the cloth and drop it in the pan.",
    "Green ice frosted the punch bowl.",
    "A stuffed chair slipped from the moving van.",
    "The stitch will serve but needs to be shortened.",
    "A thin stripe runs down the middle.",
    "A six comes up more often than a ten.",
    "Lush fern grow on the roadside.",
    "If it rains, the plan will fall through.",
    # Everyday, dictation-shaped sentences (numbers, dates, names, tone)
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
    "Could you double-check the figures on the third slide for me?",
    "We're aiming to ship the beta by the end of the quarter.",
    "I left my keys and wallet on the kitchen counter again.",
    "Let me know if two o'clock works better than three for you.",
    "The train was delayed by about twenty minutes this morning.",
    "Add milk, eggs, bread, and coffee to the shopping list.",
    "His new address is fourteen Maple Street, apartment six.",
    "The battery lasts roughly ten hours on a single charge.",
    "Please confirm your reservation at least forty-eight hours ahead.",
    "We hiked nearly eight miles before stopping for lunch.",
    "The password must contain a number and a capital letter.",
    "I'm running about five minutes late, so start without me.",
    "Send the signed contract back to me as a PDF, please.",
    "The library closes at nine on weekdays and five on Sundays.",
    "Our revenue grew twelve percent compared to last year.",
    "Don't forget to back up the database before the migration.",
    "She scored ninety-eight on the exam without much studying.",
    "The recipe calls for two cups of flour and a pinch of salt.",
    "Let's push the launch to Monday to give the team more time.",
    "I really appreciate you covering my shift on such short notice.",
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
    "Have you had a chance to look into {term} yet?",
    "The whole plan really depends on {term} being ready in time.",
]


def _term_sentences(dictionary_terms: list[str] | None, rng: random.Random) -> list[str]:
    if not dictionary_terms:
        return []
    terms = dictionary_terms[:]
    rng.shuffle(terms)
    out = []
    for i, term in enumerate(terms):
        out.append(DICTIONARY_TEMPLATES[i % len(DICTIONARY_TEMPLATES)].format(term=term))
    return out


def generate_prompts(
    dictionary_terms: list[str] | None = None,
    n: int = 20,
    seed: int | None = None,
) -> list[str]:
    """Curated prompts, interleaving dictionary-term sentences (~1 in 3).

    Fully offline and deterministic given ``seed`` — this is the fallback
    when no LLM is available.
    """
    rng = random.Random(seed)
    general = BASE_SENTENCES[:]
    rng.shuffle(general)
    term_sentences = _term_sentences(dictionary_terms, rng)

    prompts: list[str] = []
    while len(prompts) < n and (general or term_sentences):
        # every third prompt exercises a dictionary term, when available
        if term_sentences and (len(prompts) % 3 == 2 or not general):
            prompts.append(term_sentences.pop(0))
        elif general:
            prompts.append(general.pop(0))
    return prompts


def generate_llm_prompts(
    llm,
    dictionary_terms: list[str] | None = None,
    n: int = 20,
) -> list[str]:
    """Ask a local LLM for fresh reading prompts; fall back to the corpus.

    ``llm`` is any object with ``generate(system_prompt, user_prompt)`` (the
    project's :class:`~wisperfree.llm.base.LLMBackend`). Any failure — LLM
    offline, malformed output, too few lines — falls back to
    :func:`generate_prompts`, so callers always get ``n`` usable prompts.
    """
    if llm is None:
        return generate_prompts(dictionary_terms, n=n)

    terms_line = ""
    if dictionary_terms:
        terms_line = (
            " Naturally work these exact terms into some sentences "
            f"(spelled exactly like this): {', '.join(dictionary_terms[:30])}."
        )
    system = (
        "You generate sentences for a speech-recognition training session. "
        "The user will read each sentence aloud. Produce clear, natural, "
        "easy-to-read English sentences of 8 to 16 words, phonetically varied, "
        "covering everyday speech, numbers, and dates."
    )
    user = (
        f"Give me {n + 5} sentences to read aloud, one per line, no numbering, "
        f"no quotes, no commentary.{terms_line}"
    )
    try:
        raw = llm.generate(system, user)
        prompts = _parse_sentences(raw)
    except Exception:
        log.exception("LLM prompt generation failed; using curated corpus")
        return generate_prompts(dictionary_terms, n=n)

    if len(prompts) < n:
        # top up from the curated corpus without duplicating
        seen = {p.lower() for p in prompts}
        for extra in generate_prompts(dictionary_terms, n=n * 2):
            if extra.lower() not in seen:
                prompts.append(extra)
                seen.add(extra.lower())
            if len(prompts) >= n:
                break
    return prompts[:n]


_LEADING_JUNK = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def _parse_sentences(raw: str) -> list[str]:
    out: list[str] = []
    for line in raw.splitlines():
        line = _LEADING_JUNK.sub("", line).strip().strip('"').strip()
        # keep sentence-like lines only
        words = line.split()
        if 3 <= len(words) <= 40 and any(c.isalpha() for c in line):
            out.append(line)
    return out
