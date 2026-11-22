"""Drafting the text file with an open-weight model: everything that needs no model and no torch.

Why this exists. The text that the trained model reads (`text/model_text.yaml`) must not be written by a
closed model. So an open-weight model writes it, one field at a time. This module is the program around it:

    instructions   `text/drafting_instructions.yaml`: for every field what the text must contain and its form.
                   That file was written by the coding assistant (a closed API model). It holds all the words
                   that the writing model is given; this module adds only line breaks, "- ", ": " and dashes.
    writer         a callable `(messages, seed, max_new_tokens) -> text`. The real one is built in
                   `scripts/gpu/draft_text.py`; tests use fake ones.
    selection      for each field up to K candidates are asked for. Candidate i has the seed
                   `derive_seed(run_seed, field key, i)`. Each reply is cleaned by fixed rules (for the
                   two rule prompts this includes the line layout, `lay_out`, which changes no word) and
                   then checked by fixed rules, and the FIRST valid candidate in index order is taken. Nobody
                   chooses between candidates. A field with no valid candidate is reported as failed, and
                   the run goes on.
    log            one line per candidate: the messages, the raw reply, the cleaned value, valid or not and
                   why, taken or not, and the longest run of words that it shares with the shown text.

The checks are the rules of `accessdesk.text.check_text` for the field, plus the content rules listed at
`check_value` and `check_item`. The finished file is written only if `check_text` accepts it as a whole.

The copy rule. The instructions are our words, and the writer must not hand them back as its own. So every
candidate is compared with the text that its writer was shown (`shown_texts`), and a candidate that repeats a
run of `COPY_LIMIT` or more consecutive words of it is refused (`copy_problem`). The longest shared run of
every candidate, valid or not, goes into the log. `placeholder_report` makes the same measurement against
`text/placeholder_text.yaml`, which the writer never sees; that one is a report and refuses nothing.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import yaml

from accessdesk.env.rules import RESOLUTION_OF_ACTION, RULESETS, decide
from accessdesk.rng import derive_seed
from accessdesk.text import (ALL_TOOL_KEYS, CONFIRMATIONS, ERROR_KEYS, FORBIDDEN, MIN_DEPARTMENTS, MIN_ROLES,
                             MIN_TEMPLATES_PER_STYLE, RESULT_KEYS, ROOT, STYLES, TEMPLATE_VARIABLES,
                             TOOL_PARAMS, check_text, normalise, placeholder_path)
from accessdesk.types import RESOLUTIONS, STATUSES, TOOL_KEYS

INSTRUCTIONS_PATH = ROOT / "text" / "drafting_instructions.yaml"

ASK_KINDS = ("line", "text", "list")             # what the model is asked for
MAX_LINE_CHARS = 200                             # one-line values and list items
PROMPT_WORDS = (80, 400)                         # the main and the policy-shift system prompt
NAME_WORDS = (1, 4)                              # department names and role names
MAX_NEW_TOKENS = {"line": 120, "text": 900, "list": 600}
CUT_OFF_REASON = "the reply reached the token limit before its end"
SPECIAL_TOKEN_MARKS = ("<|", "|>")               # the marks of every special token, such as <|endoftext|>
FENCE = "```"                                    # a code fence that the cleaning could not remove

# The copy rule: a candidate that shares a run of this many consecutive words (or more) with the text that its
# writer was shown is refused. The number 8 is our judgement; no source gives it. The reason for it: where a
# value and its instruction have the same content, short technical phrases of four to seven words ("the id of
# the person who") cannot be avoided, so a lower limit would refuse honest values; a run of eight or more
# words is a clause that was taken over. The same limit holds for the three system prompts.
COPY_LIMIT = 8
COPY_REASON = "copies the instruction"
NOTE_ARROW = "->"                                # the mark of the notes in the instruction file
# Words that the plain system prompt may not hold (it must not tell a rule of the desk): "sensitive",
# "manager", "approve", "approval", "approver". Each is refused in any case of letters and also as the start
# of a longer word, so "sensitivity", "managers" and "approving" are refused too. "sensible", "manage" and
# "prove" are not: they do not begin with one of these.
# The last three stems are check C10 (see the block of C6 to C10 below): the plain prompt may not tell what
# the desk does with a request. They are the starts of the forms of three verbs: "giv" (give, gives, given,
# giving, giver), "grant" (grant, grants, granted, granting, grantor, grantee), "assign" (assign, assigns,
# assigned, assigning, assignment, assignee). "gave", "gift", "forgive" and "reassign" do not begin so.
ACTION_STEMS = ("giv", "grant", "assign")
PLAIN_PROMPT_STEMS = ("sensitiv", "manager", "approv") + ACTION_STEMS

# ---------------------------------------------------------------------------------------------------------
# The checks added after the calibration of 2026-11-17 (marked C1 to C5 at `check_value` and `check_item`).
# The first text was used in 768 episodes each of three untrained models (2B, 4B, 9B). Five places of the
# text were unclear. Each check below answers one of them, and the value that was in the text then fails it.
# ---------------------------------------------------------------------------------------------------------
# C1 (both rule prompts). Measured: the prompt named its tools only in one closing sentence; steps one and
# two and case c named none. The 2B never called the role-checking tool in 445 of 768 episodes, invented a
# tool 297 times and gave the role name to the people-finding tool 545 times. So a tool must be named where
# it is used. A line that names LISTING_LINE_TOOLS or more tools of the rule set besides the closing tool
# is a list of tools and does not count; every tool of the rule set must stand on a line that counts. (The
# closing tool is left out of the count because a case names it beside the tool of its own action.) And
# every case must be a line of its own: each resolution value of the rule set stands on a line that holds
# no other resolution value of the rule set.
LISTING_LINE_TOOLS = 2
# C2 (both rule prompts). Measured: "use the department from the request to decide which one is correct"
# did not say that the choice is made among the people that the tool returned; the 2B gave the department
# to the people-finding tool as a second argument in 188 episodes. So a line that names that tool, taken
# with the line after it, must hold both of these: "list" as a whole word, and a word that begins with
# "department". (The result key of the list is not accepted in place of "list": in the first text that key
# was "people", an ordinary word of the unclear sentence itself.)
LOOKUP_LINE_WORD, LOOKUP_LINE_STEM = "list", "department"
# C3 (descriptions). Measured: a parameter description said "The resolution parameter" where the visible
# name is request_resolution; the 4B wrote the parameter name "resolution" in 94 closing calls. So the word
# before "parameter" must be the visible name of the parameter (in a tool's description: of one of the
# tool's parameters), or one of these words, which nobody takes for a name. And a tool's description may
# not hold the visible name of a parameter that only other tools have, unless that word is a result key
# too (a description may say what a tool returns).
BEFORE_PARAMETER_WORDS = ("a", "an", "the", "this", "that", "each", "every", "its", "one", "single", "only",
                          "required", "first", "second", "third", "last")
_BEFORE_PARAMETER = re.compile(r"([A-Za-z0-9_]+)\s+parameter(?![A-Za-z0-9_])", re.IGNORECASE)
# C4 (request templates). Measured: "... the {role_name} access in {department}" made models send the role
# name with the word "access" at its end to the role-checking tool (4B 23 errors, 9B 28, all from this one
# template). So directly after {role_name} (closing quotation marks and brackets are skipped) only these may
# stand: the end of the template, a punctuation mark, or one of these words. No template of the first text
# had the same fault after {person_name} or {department}, or before {role_name}, so there is no rule there.
AFTER_ROLE_WORDS = ("and", "as", "asap", "at", "because", "but", "by", "can", "could", "for", "from", "has",
                    "he", "if", "in", "is", "now", "of", "on", "or", "please", "pls", "role", "she", "should",
                    "since", "so", "soon", "thanks", "that", "they", "thx", "to", "today", "was", "when",
                    "while", "will", "with", "within", "would")
_AFTER_ROLE = re.compile(r"\{role_name\}[\"'\u201d\u2019)\]]*(?:-(?=[A-Za-z0-9]))?\s*(\{|[A-Za-z0-9_]+)?")
# C5 (the prompt of a desk that never gives a role itself). Measured defect: the policy-shift prompt kept a
# fourth case that can never apply, listed the role-giving tool among the names to use, and never said that
# this tool is not called. So every sentence that names that tool must hold one of NEGATIONS as a whole
# word, there must be such a sentence, and the prompt may not speak of a fourth case. A sentence that names
# the tool may not hold a word of SECOND_CALL_WORDS either: the main prompt forbids a second giving of a
# role, and that sentence, left in, would pass with its "never" and forbid the second call only.
NEGATIONS = ("not", "never")
SECOND_CALL_WORDS = ("twice", "second")
_FOURTH_CASE = re.compile(r"\bcase\s+d\b|^\s*\(?d[.):]|\b(?:four|4)\s+cases\b|\bfourth\s+case\b",
                          re.IGNORECASE | re.MULTILINE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")               # where a line is cut into sentences

# ---------------------------------------------------------------------------------------------------------
# The checks added after the second calibration of 2026-11-17 (marked C6 to C10 at `check_value` and
# `check_item`). Six fields were written again (round 2) and used in 768 episodes each of two untrained
# models (2B, 4B). Five new faults were found. Each check below answers one of them, and the round-2 value
# fails it.
# ---------------------------------------------------------------------------------------------------------
# C6 (both rule prompts). Measured: the round-2 prompt had no sentence which says that the cases are tested
# in order and that the first case that fits is followed. On the 14 calibration families in which more than
# one rule fact is true the 4B passed 6.9% (90.9% on the other 34), and 414 of its 417 failures there
# followed another order; the 2B passed 38.6% (72.9%). So, part 1: above the first line that holds a
# resolution value of the rule set there is a sentence that holds the whole words of ORDER_LINE_WORDS and a
# word that begins with ORDER_LINE_STEM (any case of letters). Two sentences that stand one directly after
# the other on the same line count as one ("Test the cases in this order. Follow the first case that
# fits."). The sentence, or each of the two, names no tool of the desk. (Changed on 2026-11-18. Before, the
# three words could stand anywhere on one line, and a reviewer showed that a normal line of step 1 passed:
# "1. First call lookup_person in order to get the list; in case of several people use the department.")
# Part 2: the lines of the cases stand in the order of testing. For each resolution value of the rule set
# the first line that holds it is taken, and these lines must come in the order of CASE_ORDER. That is the
# order in which `rules.decide` tests the facts: account not active, role held already, role sensitive,
# none of these.
ORDER_LINE_WORDS, ORDER_LINE_STEM = ("order", "first"), "case"
CASE_ORDER = ("denied", "already", "pending", "granted")        # internal keys of values.resolution
# C7 (both rule prompts). Measured, round 2 against round 1: pass rate on families with an inactive account
# 2B 12.5% (was 50.5%), 4B 79.2% (was 95.3%); a role was given to an inactive person in 98 of 768 episodes
# (2B) and in 24 (4B). The round-2 prompt listed the three values of the role check directly before the
# cases, and its case a said "If the account_state is inactive" with no source. The first form of this
# check asked for the result key and the source on the line of case a. On the 16 candidates of run seeds
# 4 and 5 (text/drafting_log.jsonl) that was more than the writer gave: 12 failed it, and several named
# the key in the line of step 1 instead (counts of _dev/round3/draft_diagnosis.py, not checked again). A
# reviewer showed that a case a with the word for an account in use passed. So the check was changed on
# 2026-11-18. Of the same 16 candidates, 9 fail the changed check (a test counts them): 7 never hold the
# key, and 2 hold it only in case a, with no source. Two parts:
# (i) the first line that holds the resolution value meaning REFUSED holds the account word for a
# switched-off account (whole word, any case of letters) and does not hold the account word for an account
# in use (the same way).
# (ii) the visible result key of the account state, letter for letter, stands in one of two places: (A) on
# a line that names the people-finding tool or on the line directly after it (the pair of lines that check
# C2 reads); or (B) on that first line with the REFUSED value, together with a source on the same line:
# the visible name of the people-finding tool, or the word "step" followed by "1" or "one" (any case of
# letters). Such a line is no list of tools for C1: the closing tool is not counted there.
_STEP_ONE = re.compile(r"(?<![A-Za-z0-9_])step\s+(?:1|one)(?![A-Za-z0-9_])", re.IGNORECASE)
# C8 (request templates). Measured: with the round-2 template "... give {person_name} the {role_name} in
# {department}?" the 2B sent "<role> in <department>" as the role name in 80 refused calls (59 of the 80
# episodes of that template; pass rate 24%, the other eleven templates 57 to 89%); the 4B in 8 calls. So,
# if {department} is the next variable after {role_name}, a mark of ROLE_TO_DEPARTMENT_MARKS or at least
# ROLE_TO_DEPARTMENT_WORDS words must stand between the two. Changed on 2026-11-18, after a reviewer showed
# that "{role_name}-{department}", "{role_name}/{department}" and "{role_name} & {department}" passed: only
# the listed marks count now (a hyphen, a slash, an ampersand and a quotation mark do not), and the number
# of words went from 3 to 4 ("the {role_name} role in the {department}" has three and still reads as one
# name). For the count, an apostrophe or a hyphen inside a word is dropped ("person's", "in-house": one
# word each). The number 4 is our judgement; no source gives it. And the mirrored order: {role_name} may
# not stand directly after {department} with only white space between them ("of {department} {role_name}
# now" reads as one name: 6 of the 14 role names of the text begin with a department word).
ROLE_TO_DEPARTMENT_WORDS = 4
ROLE_TO_DEPARTMENT_MARKS = (".", ",", ";", ":", "(", ")", "?", "!", "|")
_DEPARTMENT_THEN_ROLE = re.compile(r"\{department\}\s*\{role_name\}")
_INSIDE_WORD_MARK = re.compile(r"(?<=\w)['\u2019-](?=\w)")
# C9 (request templates). Seen in round 2, held-out style: two of the three templates were nearly one
# sentence ("I am {person_name} from {department} and I need the {role_name} role to do my daily tasks." and
# "I am {person_name} from {department}. I need the {role_name} role to complete my daily tasks."). So a
# template is refused if it shares a run of SAME_RUN_LIMIT or more words in a row with a template of the
# same field that was accepted before it. Words are counted as the copy rule counts them (`words_of`): lower
# case, punctuation removed, a variable is one word. The number 5 is our judgement; no source gives it.
SAME_RUN_LIMIT = 5
# C10 (plain prompt). Seen in round 2: "You work at an IT desk where one request means giving a role to a
# person." This can be read as: a request is settled by giving the role. So the plain prompt may hold no
# word that begins with a stem of ACTION_STEMS (they stand in PLAIN_PROMPT_STEMS above).

# ---------------------------------------------------------------------------------------------------------
# Added on 2026-11-18, after two drafting runs (run seeds 4 and 5) in which all 16 candidates of the main
# rule prompt were refused (text/drafting_log.jsonl).
# ---------------------------------------------------------------------------------------------------------
# The line layout (both rule prompts). Measured: 14 of the 16 candidates were one paragraph with no line
# break, and checks C1, C2, C6 and C7 read lines; check C1 refused all 16. So the program lays out the
# lines itself (`lay_out`): it starts a new line in front of each sentence that begins with a marker. The
# markers: the word "Step" and a number (digits, or a word of STEP_NUMBER_WORDS in any case of letters);
# the word "Case" and one single letter from a to h (either case) that no other letter follows; the word
# "Finally". A marker counts only with a capital first letter and only at the start of a sentence: at the
# start of the text, or after one of the characters of LAYOUT_SENTENCE_ENDS and white space. The white
# space in front of the marker becomes one line break. Where the writer wrote a line break, nothing
# changes. No word and no other character is changed.
RULE_PROMPTS = ("system_prompt", "policy_shift.system_prompt")   # the fields with the form system_prompt
STEP_NUMBER_WORDS = ("one", "two", "three", "four", "five")
LAYOUT_SENTENCE_ENDS = ".!?:"
_STEP_MARK = r"Step[ \t]+(?:\d+|(?i:" + "|".join(STEP_NUMBER_WORDS) + r"))(?![A-Za-z0-9_])"
_CASE_MARK = r"Case[ \t]+[A-Ha-h](?![A-Za-z])"
_END_MARK = r"Finally(?![A-Za-z0-9_])"
_LAYOUT_BREAK = re.compile(r"(?<=[" + re.escape(LAYOUT_SENTENCE_ENDS) + r"])[ \t]+(?=" + _STEP_MARK + "|"
                           + _CASE_MARK + "|" + _END_MARK + ")")
_STEP_LINE, _CASE_LINE = re.compile(r"\s*" + _STEP_MARK), re.compile(r"\s*" + _CASE_MARK)
_MARKER_LINE = re.compile(r"\s*(?:" + _STEP_MARK + "|" + _CASE_MARK + "|" + _END_MARK + ")")
# The first check on lines (both rule prompts), so that the log tells the true cause of a prompt without
# lines: after the layout, at least MIN_STEP_LINES lines begin with a step marker, and one line for each
# resolution value of the rule set begins with a case marker (4, or 3 at a desk that never gives a role).
MIN_STEP_LINES = 2
# C11 (the resolution value that means: approval was requested, the role is not given, the request waits
# for the approver). The kept value was "request_closing". It does not tell that meaning and reads like a
# name for the act of closing (a reviewer's finding on the round-2 text; no effect was measured). So the
# value, cut into its words at the underscores, must hold a word that begins with a stem of PENDING_STEMS.
PENDING_STEMS = ("pend", "approv", "wait", "await")

# ---------------------------------------------------------------------------------------------------------
# The checks added after the third calibration of 2026-11-18 (C12 to C20). The round-3 text (SHA-256
# 1ebc268f825c9a15d5ccabece35da993e4b4bfb0507a7f6d92938b3452a4f221) was used in 1536 episodes each of two
# untrained models (2B, 4B), on the harder design data/scenarios_calib3_mixed. Nine faults were found. The
# numbers below are those of _dev/round3/second_try_report.json, as they were given to this change; they
# were not counted again here. Each check answers one fault, and the round-3 value fails it.
# How these checks read a rule prompt (`marker_lines`): a line runs from one marker of the layout to the
# next. A line of the text that begins with no marker and follows no empty line belongs to the line above
# it. So a step or a case that the writer wrapped over two lines is read as one.
# ---------------------------------------------------------------------------------------------------------
# C12, fact form (both rule prompts). Measured: step 1 ended with "Show the department, the
# account_state, and the manager_id for each person in the list." The notes meant a fact about the tool
# result, and the writer made an order of it. Both models then wrote these values as words in front of
# their calls: well-formed calls with words in front went from 1.2% to 44.5% (4B) and from 3.0% to 11.0%
# (2B), and messages cut off at the token cap went from 37 to 149 (2B). So, in the line of step 1, each
# sentence that holds the result key of the account state or the result key of the manager id must begin
# (after a marker such as "Step 1:") with a word of FACT_FIRST_WORDS, in any case of letters. An order
# begins with a verb or with "Then", and those are not in the list. The name of the people-finding tool
# may begin such a sentence too ("lookup_person returns a list ..." states a fact).
FACT_FIRST_WORDS = ("the", "each", "every", "for", "in", "it", "this", "that", "these", "those", "all",
                    "both", "its", "their", "there", "a", "an", "any")
# C13, whose (both rule prompts). Measured fault of the text: case a said "the account_state" and case c
# "the manager_id", and did not say whose. With two people of one name the list holds two states and two
# managers. (No count was taken for this fault alone.) So the first line with the value meaning REFUSED
# and the first line with the value meaning WAITS FOR APPROVAL each hold a word that ties the state or the
# manager to the person: "person" as a whole word in any case of letters ("person's" counts), or "step"
# followed by "1" or "one", or the name of the people-finding tool.
WHOSE_WORD = "person"
# C14, two messages (both rule prompts). Measured: cases c and d named two calls each, with no word that
# says "one after the other". The 4B sent two calls in one message (115 messages; round 2: 50), the
# message was refused, and it then closed as given with no giving (33 episodes). The 2B closed as waiting
# for approval and never asked for approval (75 episodes; round 2: 39). So, on the first line with the
# value meaning WAITS FOR APPROVAL that names the approval tool, and on the first line with the value
# meaning GIVEN that names the role-giving tool: the closing tool is named after that tool, and a word of
# SEQUENCE_WORDS or a phrase of SEQUENCE_PHRASES stands between the first place of that tool's name and
# the end of the sentence that first names the closing tool. If the line does not name the closing tool,
# the resolution value stands in its place. (The word "Finally" is no such word: a sentence that begins
# with it starts a new line by the layout.)
SEQUENCE_WORDS = ("then", "next", "after", "afterwards", "once", "later", "following")
SEQUENCE_PHRASES = ("new message", "next message", "separate message", "another message", "second message")
# C15, keys in the cases (both rule prompts). Measured: case b said "If the person already holds the role"
# and case c "If the role is sensitive", with no result key (round 2 had "If the holds_role value is
# true"). The 2B gave a role that was held already in 55 of 384 episodes (round 2: 8). So the first line
# with the value meaning HELD BEFORE holds the result key for "holds the role", letter for letter; and,
# under the default rule set only, the first line with the value meaning WAITS FOR APPROVAL holds the
# result key for "sensitive".
# C16, step 2 for every request (both rule prompts). Measured: no sentence said that step 2 is done also
# when the account is switched off, and only step 2 shows the role id that the closing call needs. The 2B
# never called the role-checking tool in 123 episodes (round 2: 77). So the line of step 2 holds a word
# of EVERY_REQUEST_WORDS or a phrase of EVERY_REQUEST_PHRASES.
EVERY_REQUEST_WORDS = ("always", "every", "even", "regardless", "whatever")
EVERY_REQUEST_PHRASES = ("no matter", "all cases", "each case", "any case", "all requests", "each request")
# C17, the two working rules (both rule prompts: the lines above the first step line; the plain prompt:
# anywhere). Measured: one candidate of the main prompt (run seed 6) passed every check and held neither
# rule. And the round-3 plain prompt joined the two rules: "You must use exactly one tool call per message
# to find the correct id shown in the result. Any other id is not allowed." So: (i) one sentence holds a
# word of ONE_CALL_WORDS, a word that begins with CALL_STEM and a word that begins with a stem of
# MESSAGE_STEMS; (ii) one sentence holds a word of ID_WORDS, a word that begins with a stem of
# RESULT_STEMS and a word of ONLY_WORDS. ("return" is accepted beside "result": "ids that a tool has
# returned" says the same.)
ONE_CALL_WORDS, CALL_STEM, MESSAGE_STEMS = ("one", "single"), "call", ("message", "repl", "turn", "respons")
ID_WORDS, RESULT_STEMS, ONLY_WORDS = ("id", "ids"), ("result", "return"), ("only", "never", "not", "no")
# C18, the second desk (the prompt of a desk that never gives a role itself; part iv: both rule prompts).
# Seen in the round-3 text: the prompt of the second desk still said "Never give a role twice", said "the
# account is in use" where the tool result says "active", and ended with a list of four tools that left
# out the role-giving tool, which stands in the model's list of tools. So: (i) no sentence holds a word
# of TWICE_WORDS or a phrase of TWICE_PHRASES together with a word that begins with a stem of
# ACTION_STEMS ("second message" is not counted); (ii) if the first line with the value meaning WAITS FOR
# APPROVAL holds "account" as a whole word or the result key of the account state, it also holds the
# account word for an account in use; (iii) the value meaning GIVEN stands only in a sentence that holds a
# word of UNUSED_WORDS; (iv) a sentence that names TOOL_LIST_SIZE or more of the five tools and holds no
# resolution value names all five.
# "at most once" and "only once" are in the list because the notes of the first desk now say that a role
# is given once at most, and the prompt of the second desk starts from the prompt of the first desk.
TWICE_WORDS = ("twice", "again", "second")
TWICE_PHRASES = ("two times", "more than once", "at most once", "only once")
UNUSED_WORDS = ("never", "not", "no")
TOOL_LIST_SIZE = 3
# C19, plain prompt. The round-3 value had 42 words, and its form said at most 40.
PLAIN_MAX_WORDS = 40
# C20, value names (the resolution values). Measured: the value for a role that was held already was
# "no_change", which is also true of a refused request. Episodes with a switched-off account that closed
# with this value: 2B 134 of 384 (round 2, value role_already_held: 60); 4B 108 of the 128 episodes in
# which the role was held as well (the report file gives no round-2 number for these). So, cut at
# the underscores: the value meaning HELD BEFORE holds a word that begins with a stem of ALREADY_STEMS;
# the value meaning REFUSED holds a word that begins with a stem of DENIED_STEMS; and no resolution value
# holds a word that begins with a stem of VAGUE_STEMS (words that fit more than one ending).
ALREADY_STEMS = ("alread", "held", "hold", "has", "have", "exist", "own")
DENIED_STEMS = ("den", "refus", "reject", "declin", "block", "inactiv", "disabl")
VAGUE_STEMS = ("chang", "done", "clos", "ok", "complet", "finish", "success", "fail", "error", "resolv",
               "updat", "process")

# ---------------------------------------------------------------------------------------------------------
# The check added after text round 4 of 2026-11-18 (C21). The round-4 text (SHA-256
# 71e18ceab0239b91318f3ec00957afe2f10e5c9fae9ca1413dc2d21314643b3d) passed checks C1 to C20, and its
# second prompt was wrong. The fault was found by reading (_dev/round4/text_ready.flag). No effect of it
# on a model was measured.
# ---------------------------------------------------------------------------------------------------------
# C21, the last case of the second desk (the prompt of a rule set that asks for approval and under which
# no decision depends on whether the role is sensitive; today that is all_grants_need_approval only).
# Seen in the round-4 text: case c of the second desk said "IF the value of is_sensitive is true", which
# is the condition of case c of the first desk. At the second desk case c is for every request that
# cases a and b did not close, sensitive or not (`rules.decide`). With that text a role that is not
# sensitive fits no case. The field was drawn in run seeds 10, 11 and 12 (24 candidates in
# text/drafting_log.jsonl), and both candidates that were taken had this condition. Of the 24, 15 hold
# the result key for a sensitive role in the line of case c (a test counts them). No check read that
# line for it. The first form of this check read only the first line with the value meaning WAITS FOR
# APPROVAL, and only for that key and for SENSITIVE_STEM. A reviewer then showed wrong prompts that
# passed (_dev/round4/c21_report.json): the key in another line, another word for a kind of role
# ("IF the role is restricted"), and a fourth case with another mark ("Case e", "Case 4", "Case c2").
# So the check has three parts now:
# (i) the flag. The result key for a sensitive role and a word that begins with SENSITIVE_STEM (both in
# any case of letters, as check C5 reads the name that it forbids) stand in the line of step 2 only. The
# result of the role check holds that key, so step 2 names it. Every other marker line is read. A line
# of case c that says "sensitive or not" is refused too. That sentence is true at this desk, but a fixed
# rule cannot tell it from a false condition, and the line does not need it.
# (ii) three case lines. The marker lines that begin with the word Case carry the marks a, b and c, each
# one time. The word "case" with another mark behind it (one letter, a number, or a letter and a number:
# "Case e", "case 4", "Case c2") stands nowhere. And for each resolution value of the rule set, the first
# line that holds it begins with the marker of its own case. That is how a case is seen that an empty
# line cuts in two: its second part holds the value and begins with no marker.
# (iii) the condition of case c. The condition is the text of the line of case c between its marker and
# the first of: the word "then" (any case of letters), the first name of a tool, or, if the line has no
# "then", the first sentence end after the word "if". With none of these it is the first sentence. It
# may hold only: words of LAST_CONDITION_WORDS; the account word for an account in use, and with it the
# result key of the account state; the result key for "holds the role" together with "false" or "not".
# Every other word is refused, so no word for a kind of role can stand there, and "true" cannot either.
# The list is our judgement. Its first 36 words were given with the task. The other 16 are words that a
# right condition of the tests needs ("the person does not yet hold the role" stood in a prompt that the
# writer drew for this desk in the first text round; held, already, regardless and whatever came with
# the second review, which is told below). The word
# "role" is in the list, and no word for a kind of role is. Two more rules, because the list alone
# lets a turned-round condition pass: "not" or "no" does not stand directly in front of the account
# word for an account in use or of "false"; and a condition with a word of FIT_WORDS ("IF all of the
# above cases fit", "IF the person does hold the role") also holds a word of NOT_WORDS. And behind the
# condition the line holds no
# word of SECOND_CONDITION_WORDS ("THEN, if the role is restricted, call ..."). Of the 24 candidates, 9
# pass part iii: those with "IF the account_state of the person is active" (a test counts them).
# A second reviewer then showed three more kinds of wrong prompts that passed. No candidate of a run had
# one of them; the repairs are our judgement.
# (1) A condition made only of words of the list: "IF the role is a manager role", "IF the request has
# approval". The list held words of the action (call, ask, manager, approval, with), because in a line
# without "then" the condition ran up to the first tool name. Now the condition also ends at the first
# word of ACTION_WORDS that begins a clause (it stands at the start, or behind a comma, a colon, a
# semicolon or a sentence end), and those five words are out of the list. The words role, hold and held
# stand only in one of the forms of _ROLE_TALK ("does not hold the role", "the role is not held yet",
# "for any role"), so "IF the person has the role" and "does hold the role" are refused. And "not" or "no"
# is refused also with one small word between it and the account word ("not an active account").
# (2) A kind of role in words that are neither the key nor the stem ("Only restricted roles reach case
# c"). A new part (iv), which runs last, refuses a word that begins with one of ROLE_KIND_STEMS, in every
# line but the line of step 2. In the line of step 2, a sentence that holds the key, the stem or such a
# word holds no word of STEP_2_NOT_WITH and no resolution value: that line tells what the result holds,
# not what to do.
# (3) A limit behind the action of case c ("for restricted roles only"). The kind word is refused by
# (iv), and "only" is one of SECOND_CONDITION_WORDS now. The words "even if" are not a second condition.
SENSITIVE_STEM = "sensitiv"
# Words for a kind of role, as stems (our judgement; a test pins them). At the second desk every role is
# handled in the same way, so its prompt needs none of them.
ROLE_KIND_STEMS = ("restrict", "unrestrict", "nonrestrict", "privileg", "unprivileg", "flag", "unflag",
                   "special", "critical", "protected", "unprotected", "elevated", "confidential", "admin",
                   "ordinary", "normal", "regular", "risk", "secret", "classified")
STEP_2_NOT_WITH = ("approv", "case", "only", "manager")         # stems; see (2) above
ACTION_WORDS = ("call", "ask", "request", "close", "give", "record", "use", "send", "do")
# Part (v), from the same review: "Do not call grant_role until the manager approves" passed check C5 and
# says that the role is given later. A sentence with the name of the tool that gives a role holds no word
# of LATER_WORDS. The words "even after", "even before" and "even if" are not counted.
LATER_WORDS = ("until", "till", "before", "after", "unless")
CASE_MARKS = "abcd"                                             # the marks of the cases, in CASE_ORDER
LAST_CONDITION_WORDS = (
    "if", "when", "none", "no", "not", "of", "the", "above", "before", "earlier", "previous", "other",
    "case", "cases", "a", "b", "and", "or", "do", "does", "did", "fit", "fits", "applies", "apply",
    "every", "any", "all", "in", "otherwise", "else", "is", "has", "value", "person", "account",
    "an", "neither", "nor", "for", "request", "requests", "that", "reaches", "this",
    "yet", "hold", "role", "held", "already", "regardless", "whatever")
FIT_WORDS = ("fit", "fits", "applies", "apply", "hold")
ROLE_WORDS = ("role", "hold", "held")                           # they stand only in a form of _ROLE_TALK
SECOND_CONDITION_WORDS = ("if", "unless", "except", "only")     # behind the condition: a second condition
NOT_WORDS = ("none", "no", "not", "neither", "nor", "other")
_CASE_HEAD = re.compile(r"\s*Case(?![A-Za-z0-9_])[\s:.)\-]*([A-Za-z0-9_]*)")   # a line that begins with Case
_CASE_NAMED = re.compile(r"(?<![A-Za-z0-9_])case[ \t]+([A-Za-z]\d*|\d+)(?![A-Za-z0-9_])", re.IGNORECASE)
_THEN = re.compile(r"(?<![A-Za-z0-9_])then(?![A-Za-z0-9_])", re.IGNORECASE)
_IF = re.compile(r"(?<![A-Za-z0-9_])if(?![A-Za-z0-9_])", re.IGNORECASE)
_EVEN_IF = re.compile(r"(?<![A-Za-z0-9_])even\s+(?:if|after|before)(?![A-Za-z0-9_])", re.IGNORECASE)
_ACTION = re.compile(r"(?:^|[,:;.!?])\s*(" + "|".join(ACTION_WORDS) + r")(?![A-Za-z0-9_])", re.IGNORECASE)
_KIND = re.compile(r"(?<![A-Za-z0-9_])(?:" + "|".join(ROLE_KIND_STEMS) + r")[A-Za-z]*", re.IGNORECASE)
# The forms in which a condition of case c may speak of the role (its words in lower case, one space apart).
_ROLE_TALK = re.compile(r"\bnot (?:(?:yet|already) )?hold (?:the|this|that) role(?: yet)?\b"
                        r"|\b(?:the|this|that) role is not (?:(?:yet|already) )?held(?: yet)?\b"
                        r"|\b(?:any|every) role\b|\b(?:whatever|regardless of) the role\b")

IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")              # the pattern of accessdesk/text.py (tested)
ID_PREFIX = re.compile(r"[A-Za-z][A-Za-z0-9]{0,5}[_-]?")
_LIST_MARKER = re.compile(r"(?:[-*\u2022]|\d{1,3}[.)])\s+")     # tried at the start of a line
_VARIABLE = re.compile(r"\{[^{}]*\}")
# A word, as the copy rule counts words: a variable of a template or of an error text, or a run of letters,
# digits and underscores.
_WORD = re.compile(r"\{(?:" + "|".join(TEMPLATE_VARIABLES + ("value",)) + r")\}|\w+")
_APOSTROPHE = re.compile(r"(?<=\w)['\u2019](?=\w)")             # inside a word: "person's"
_QUOTE_PAIRS = (('"', '"'), ("'", "'"), ("`", "`"), ("\u201c", "\u201d"), ("\u2018", "\u2019"))
_RULE = "-" * 10                                 # the line of dashes around a value of several lines

Value = str | list[str]
Writer = Callable[[list[dict], int, int], str]


class InstructionError(ValueError):
    """The instruction file is incomplete or inconsistent. The message lists every problem."""


class DraftError(ValueError):
    """The run cannot start as asked (an unknown field, or a value that the log should hold and does not)."""


class ReplyCutOff(Exception):
    """A writer raises this when a reply reached the token limit before the end of the turn."""

    def __init__(self, text: str) -> None:
        super().__init__(CUT_OFF_REASON)
        self.text = text


# ---------------------------------------------------------------------------
# The fields of the text file
# ---------------------------------------------------------------------------
def tool_name_key(tool: str) -> str:
    return f"tools.{tool}.name"


def text_fields() -> tuple[list[str], list[str]]:
    """(string fields, list fields) of the text file, as dotted keys, in the order of the file.

    `meta.written_by` and `meta.placeholder` are left out: the program sets them.
    """
    strings = ["system_prompt", "plain_system_prompt", "policy_shift.variant", "policy_shift.system_prompt",
               "id_prefixes.person", "id_prefixes.role"]
    for tool in ALL_TOOL_KEYS:
        strings += [tool_name_key(tool), f"tools.{tool}.description"]
        for param in TOOL_PARAMS[tool]:
            strings += [f"tools.{tool}.params.{param}.name", f"tools.{tool}.params.{param}.description"]
    strings += [f"result_keys.{key}" for key in RESULT_KEYS]
    strings += [f"values.status.{key}" for key in STATUSES]
    strings += [f"values.resolution.{key}" for key in RESOLUTIONS]
    strings += [f"values.confirmation.{key}" for key in CONFIRMATIONS]
    strings += [f"errors.{key}" for key in ERROR_KEYS]
    lists = [f"request_templates.style_{style}" for style in STYLES] + ["departments", "roles"]
    return strings, lists


def _is_param_name(key: str) -> bool:
    parts = key.split(".")
    return len(parts) == 5 and parts[0] == "tools" and parts[4] == "name"


def _is_description(key: str) -> bool:
    return key.startswith("tools.") and key.endswith(".description")


def _param_name_keys(tool: str | None = None) -> tuple[str, ...]:
    """The fields that hold the visible parameter names of one tool, or of all six."""
    return tuple(f"tools.{name}.params.{param}.name" for name in ALL_TOOL_KEYS for param in TOOL_PARAMS[name]
                 if tool in (None, name))


def _resolution_keys() -> tuple[str, ...]:
    return tuple(f"values.resolution.{key}" for key in RESOLUTIONS)


def _status_keys() -> tuple[str, ...]:
    return tuple(f"values.status.{key}" for key in STATUSES)


def ruleset_tools(ruleset: str) -> list[str]:
    """The tools (internal keys) that a correct episode under a rule set can need, in TOOL_KEYS order."""
    actions = _actions(ruleset)
    wanted = {"lookup_person", "check_access", "close_request"}
    wanted |= {"request_approval"} if "approval" in actions else set()
    wanted |= {"grant_role"} if "grant" in actions else set()
    return [tool for tool in TOOL_KEYS if tool in wanted]


def ruleset_resolutions(ruleset: str) -> list[str]:
    """The resolutions (internal keys) that a rule set can produce, in RESOLUTIONS order."""
    produced = {RESOLUTION_OF_ACTION[action] for action in _actions(ruleset)}
    return [key for key in RESOLUTIONS if key in produced]


def _actions(ruleset: str) -> set[str]:
    flags = (True, False)
    return {decide(active, holds, sensitive, ruleset)
            for active in flags for holds in flags for sensitive in flags}


def _sensitive_decides(ruleset: str) -> bool:
    """True if, under the rule set, a decision depends on whether the role is sensitive (check C21)."""
    flags = (True, False)
    return any(decide(active, holds, True, ruleset) != decide(active, holds, False, ruleset)
               for active in flags for holds in flags)


# The three result keys that checks C12 and C15 read in a rule prompt, beside the key of the account state.
_CASE_KEYS = ("result_keys.manager_id", "result_keys.sensitive", "result_keys.has_access")


def validation_needs(key: str) -> tuple[str, ...]:
    """The fields whose values the check of `key` reads. They must be written before `key`."""
    six = tuple(tool_name_key(tool) for tool in ALL_TOOL_KEYS)
    if _is_description(key) and len(key.split(".")) == 3:           # a tool's description (check C3)
        return six + _param_name_keys() + tuple(f"result_keys.{name}" for name in RESULT_KEYS)
    if _is_description(key):                                        # a parameter's description (check C3)
        return six + (key.rsplit(".", 1)[0] + ".name",)
    if key == "system_prompt":                       # the result key and the two account words: check C7
        return six + _resolution_keys() + ("result_keys.status",) + _status_keys() + _CASE_KEYS
    if key == "policy_shift.system_prompt":
        return six + _resolution_keys() + ("result_keys.status",) + _status_keys() + _CASE_KEYS + (
            "policy_shift.variant", "system_prompt")
    if key == "plain_system_prompt":
        return _resolution_keys() + ("system_prompt",) + _status_keys() + six
    if key.startswith("errors."):
        return _resolution_keys()
    return ()


# ---------------------------------------------------------------------------
# The instruction file
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Field:
    key: str                        # dotted key of the field in the text file
    kind: str                       # "line", "text" or "list" (asked from the model), "fixed" or "copy"
    label: str                      # how the value is named when a later instruction shows it
    instruction: str = ""           # asked fields: the instruction, followed by its note on form
    uses: tuple[str, ...] = ()      # asked fields: earlier fields whose values are shown with the instruction
    want: int = 0                   # list fields: the number of items that are kept
    value: str = ""                 # fixed fields: the value
    source: str = ""                # fixed fields: where the value comes from. Copy fields: the field copied.

    @property
    def asked(self) -> bool:
        return self.kind in ASK_KINDS

    def depends_on(self) -> tuple[str, ...]:
        """Every field that must have a value before this one can be written."""
        copied = (self.source,) if self.kind == "copy" else ()
        return tuple(dict.fromkeys(self.uses + validation_needs(self.key) + copied))


@dataclass(frozen=True)
class Instructions:
    path: str
    sha256: str                     # hash of the raw file bytes
    system_message: str             # the system message of every request to the writer
    desk: str                       # the description of the desk that opens every user message
    task_heading: str
    earlier_heading: str
    sensitivity_words: tuple[str, ...]
    fields: dict[str, Field]        # in the order of writing

    def order(self) -> list[str]:
        return list(self.fields)

    def copies_of(self, key: str) -> list[str]:
        return [f.key for f in self.fields.values() if f.kind == "copy" and f.source == key]


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that refuses a key written twice in one mapping (plain YAML silently keeps the last one)."""

    def construct_mapping(self, node, deep=False):  # noqa: ANN001 (signature of the base class)
        keys = [self.construct_object(key_node, deep=deep) for key_node, _ in node.value]
        twice = sorted({str(key) for key in keys if keys.count(key) > 1})
        if twice:
            problem = f"keys written twice: {twice}"
            raise yaml.constructor.ConstructorError(None, None, problem, node.start_mark)
        return super().construct_mapping(node, deep)


def load_instructions(path: str | Path = INSTRUCTIONS_PATH) -> Instructions:
    """Read and check the instruction file. Raises InstructionError, which lists every problem."""
    raw = Path(path).read_bytes()
    try:
        data = yaml.load(raw.decode("utf-8"), Loader=_UniqueKeyLoader)
        writer, entries = data["writer"], data["fields"]
        forms, groups = data.get("forms", {}), data.get("groups", {})
        words = tuple(str(word) for word in data["sensitivity_words"])
        head = {name: str(writer[name]).strip()
                for name in ("system_message", "desk", "task_heading", "earlier_heading")}
    except (yaml.YAMLError, UnicodeDecodeError, KeyError, TypeError) as exc:
        raise InstructionError(f"{path} cannot be read: {type(exc).__name__}: {exc}") from exc
    problems: list[str] = [f"writer.{name} is empty" for name, value in head.items() if not value]
    fields: dict[str, Field] = {}
    for key, entry in entries.items():
        field = _read_field(str(key), entry, forms, groups, fields, problems)
        if field is not None:
            fields[field.key] = field
    problems += _file_problems(fields, words)
    if problems:
        listed = "\n".join(f"- {problem}" for problem in problems)
        raise InstructionError(f"{path}: {len(problems)} problem(s):\n{listed}")
    return Instructions(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), sensitivity_words=words,
                        fields=fields, **head)


def _read_field(key: str, entry: object, forms: dict, groups: dict, earlier: dict[str, Field],
                problems: list[str]) -> Field | None:
    """One entry of `fields` as a Field, or None (with the reason added to `problems`)."""
    if not isinstance(entry, dict) or sum(name in entry for name in ("ask", "fixed", "same_as")) != 1:
        problems.append(f"{key} must have exactly one of ask, fixed, same_as")
        return None
    if "same_as" in entry:
        source = earlier.get(str(entry["same_as"]))
        if source is None or source.kind == "copy" or not (_is_param_name(key) and _is_param_name(source.key)
                                                           and key.split(".")[3] == source.key.split(".")[3]):
            problems.append(f"{key}: same_as must name an earlier parameter-name field of the same parameter")
            return None
        return Field(key=key, kind="copy", label=source.label, source=source.key)
    label = str(entry.get("label", "")).strip()
    if not label:
        problems.append(f"{key} has no label")
    if "fixed" in entry:
        value, source = str(entry["fixed"]), " ".join(str(entry.get("source", "")).split())
        if not value.strip() or not source:
            problems.append(f"{key}: a fixed field needs a value and a source")
        return Field(key=key, kind="fixed", label=label, value=value, source=source)
    kind, form = str(entry["ask"]), entry.get("form")
    instruction = str(entry.get("instruction", "")).strip()
    if kind not in ASK_KINDS or not instruction or (form is not None and form not in forms):
        problems.append(f"{key}: ask must be one of {ASK_KINDS}, with an instruction and a known form")
    if form in forms:
        instruction += "\n\n" + str(forms[form]).strip()
    uses = _expand_uses(key, entry.get("uses", []), groups, earlier, problems)
    return Field(key=key, kind=kind, label=label, instruction=instruction, uses=uses,
                 want=int(entry.get("want", 0)))


def _expand_uses(key: str, uses: object, groups: dict, earlier: dict[str, Field],
                 problems: list[str]) -> tuple[str, ...]:
    """`uses` as field keys. A group stands for those of its members that were written before `key`."""
    found: list[str] = []
    for name in uses if isinstance(uses, list) else [uses]:
        if name in groups:
            found += [member for member in groups[name] if member in earlier]
        elif name in earlier:
            found.append(name)
        else:
            problems.append(f"{key} uses {name!r}, which is neither a group nor a field written before it")
    return tuple(dict.fromkeys(found))


def _file_problems(fields: dict[str, Field], words: tuple[str, ...]) -> list[str]:
    """Problems of the file as a whole: the set of fields, the order, the list sizes, the fixed tool names."""
    strings, lists = text_fields()
    problems = [f"{key} is not a field of the text file" for key in fields if key not in strings + lists]
    problems += [f"{key} has no entry" for key in strings + lists if key not in fields]
    minimum = {f"request_templates.style_{style}": MIN_TEMPLATES_PER_STYLE for style in STYLES}
    minimum.update({"departments": MIN_DEPARTMENTS, "roles": MIN_ROLES})
    order = list(fields)
    for key, field in fields.items():
        if field.asked and (field.kind == "list") != (key in lists):
            problems.append(f"{key}: ask must be list for a list field, and only for a list field")
        if field.kind == "list" and field.want < minimum.get(key, 1):
            problems.append(f"{key}: want must be at least {minimum.get(key, 1)}")
        late = [name for name in field.depends_on() if name not in order[:order.index(key)]]
        if late:
            problems.append(f"{key} needs {late}, which must stand above it")
    for tool in TOOL_KEYS:
        if tool_name_key(tool) in fields and fields[tool_name_key(tool)].kind != "fixed":
            problems.append(f"{tool_name_key(tool)} must be fixed: the five tool names are not asked for")
    variant = fields.get("policy_shift.variant")
    if variant is not None and (variant.kind != "fixed" or variant.value not in RULESETS[1:]):
        problems.append(f"policy_shift.variant must be fixed to one of {RULESETS[1:]}")
    if not words or any(word != word.casefold().strip() or not word for word in words):
        problems.append("sensitivity_words must be a list of lower-case words")
    return problems


# ---------------------------------------------------------------------------
# The messages of one request to the writer
# ---------------------------------------------------------------------------
def build_messages(ins: Instructions, key: str, chosen: dict[str, Value]) -> list[dict]:
    """The system message and the user message for one field.

    The user message is: the description of the desk, the instruction of the field, and the earlier values
    that the instruction uses. Raises KeyError if one of those values is not in `chosen`.
    """
    field = ins.fields[key]
    if not field.asked:
        raise ValueError(f"{key} is not asked from the model")
    parts = [_task_text(ins, field)]
    if field.uses:
        shown = [_shown(ins.fields[name].label, chosen[name]) for name in field.uses]
        parts.append(ins.earlier_heading + "\n" + "\n".join(shown))
    return [{"role": "system", "content": ins.system_message},
            {"role": "user", "content": "\n\n".join(parts)}]


def _task_text(ins: Instructions, field: Field) -> str:
    """The start of the user message: the desk, then the heading of the task with the instruction."""
    return f"{ins.desk}\n\n{ins.task_heading}\n{field.instruction}"


def _shown(label: str, value: Value) -> str:
    if isinstance(value, list):
        return f"- {label}:\n" + "\n".join(f"    {item}" for item in value)
    if "\n" in value:
        return f"- {label}:\n{_RULE}\n{value}\n{_RULE}"
    return f"- {label}: {value}"


def candidate_seed(run_seed: int, key: str, index: int) -> int:
    """The seed of candidate `index` of a field."""
    return derive_seed(run_seed, key, index)


# ---------------------------------------------------------------------------
# The copy rule: how much of the shown text a value repeats (fixed rules; no judgement)
# ---------------------------------------------------------------------------
def words_of(text: str) -> list[str]:
    """The words of a text, in the form in which the copy rule compares them.

    Lower case. An apostrophe inside a word is dropped ("person's" is the one word "persons"). Every other
    character that is not a letter, a digit or an underscore divides words and is itself no word, so
    "role-id" is two words and "lookup_person" is one. The variables {person_name}, {department},
    {role_name} and {value} are one word each, and differ from the plain words "department" and "value".
    """
    return _WORD.findall(_APOSTROPHE.sub("", text.casefold()))


def shown_texts(ins: Instructions, key: str) -> list[str]:
    """The texts of ours that the writer of field `key` is shown. A shared run lies inside one of them.

    They are: the system message; the start of the user message as one text (the description of the desk,
    the heading of the task, the instruction of the field with its note on form); and, for a field that is
    shown earlier values, the heading of that block and the label of each value.
    The earlier VALUES are not among them: a text that repeats a fixed name (or, in the second system
    prompt, the wording of the first) uses that value as it was told to.
    """
    field = ins.fields[key]
    block = [ins.earlier_heading] + [ins.fields[name].label for name in field.uses] if field.uses else []
    return [ins.system_message, _task_text(ins, field)] + block


def longest_shared_run(words: list[str], texts: Iterable[list[str]]) -> list[str]:
    """The longest run of consecutive words of `words` that also stands, in one piece, in one of `texts`.

    Of several runs of the same length the first one found is returned (texts in their order, then `words`
    from left to right). An empty list: no word is shared.
    """
    best: list[str] = []
    for text in texts:
        places: dict[str, list[int]] = {}
        for j, word in enumerate(text):
            places.setdefault(word, []).append(j)
        for i, word in enumerate(words):
            for j in places.get(word, ()):
                if i and j and words[i - 1] == text[j - 1]:
                    continue                                   # inside a run that began earlier
                n = 1
                while i + n < len(words) and j + n < len(text) and words[i + n] == text[j + n]:
                    n += 1
                if n > len(best):
                    best = words[i:i + n]
    return best


def shared_run(ins: Instructions, key: str, value: str) -> list[str]:
    """The longest run of words that `value` shares with what the writer of `key` was shown."""
    return longest_shared_run(words_of(value), [words_of(text) for text in shown_texts(ins, key)])


def run_entry(run: list[str]) -> dict:
    """A shared run as the log holds it: its length and its words."""
    return {"length": len(run), "words": " ".join(run)}


def copy_problem(ins: Instructions, key: str, value: str) -> str | None:
    """The copy rule: why `value` is refused as a copy of the text shown to its writer, or None."""
    run = shared_run(ins, key, value)
    if len(run) < COPY_LIMIT:
        return None
    return (f"{COPY_REASON}: it shares a run of {len(run)} words with the text that the writer was shown "
            f"({' '.join(run)!r}); the limit is {COPY_LIMIT}")


# ---------------------------------------------------------------------------
# Cleaning a reply (fixed rules; no judgement)
# ---------------------------------------------------------------------------
def clean_reply(kind: str, raw: str) -> Value:
    """The value that a raw reply stands for.

    Every kind: line ends made uniform; white space at both ends removed; a code fence around the whole reply
    removed; one pair of quotation marks around the whole reply removed.
    "line": then the first line that is not empty, again without white space and without quotation marks.
    "text": then white space at the end of every line removed.
    "list": then one item per line that is not empty, each without a list marker ("-", "*", "1.", "1)"),
            without white space at both ends and without quotation marks.
    """
    text = _strip_quotes(_strip_fence(raw.replace("\r\n", "\n").replace("\r", "\n").strip()))
    lines = text.split("\n")
    if kind == "line":
        first = next((line.strip() for line in lines if line.strip()), "")
        return _strip_quotes(first)
    if kind == "text":
        return "\n".join(line.rstrip() for line in lines).strip()
    if kind == "list":
        items = [_strip_quotes(_without_marker(line.strip())) for line in lines]
        return [item for item in items if item]
    raise ValueError(f"unknown kind {kind!r}")


def _without_marker(line: str) -> str:
    """`line` without a list marker at its start."""
    found = _LIST_MARKER.match(line)
    return line if found is None else line[found.end():].strip()


def _strip_fence(text: str) -> str:
    """`text` without a code fence that surrounds all of it."""
    lines = text.split("\n")
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    if len(lines) == 1 and len(text) >= 6 and text.startswith("```") and text.endswith("```"):
        return text[3:-3].strip()
    return text


def _strip_quotes(text: str) -> str:
    """`text` without one pair of quotation marks around all of it.

    The pair is removed only if the marks inside it are balanced. So `"a "b" c"` loses its outer pair, and
    `'it's'` (an apostrophe inside) is left as it is.
    """
    for opening, closing in _QUOTE_PAIRS:
        if len(text) >= 2 and text[0] == opening and text[-1] == closing:
            inner = text[1:-1]
            if opening == closing:
                balanced = inner.count(opening) % 2 == 0
            else:
                balanced = inner.count(opening) == inner.count(closing)
            return inner.strip() if balanced else text
    return text


def lay_out(text: str) -> str:
    """The line layout of a rule prompt: a new line in front of each sentence that begins with a marker.

    The markers and the measured reason are told at RULE_PROMPTS. Only white space in front of a marker is
    changed, into one line break, so every word stays as the writer wrote it. A marker that already begins
    a line is left alone, so laying out a text twice changes nothing more.
    """
    return _LAYOUT_BREAK.sub("\n", text)


def cleaned_value(field: Field, raw: str) -> Value:
    """The value that a raw reply stands for, as the checks read it, the log holds it ("cleaned") and the
    text file gets it: the cleaned reply, and for the two rule prompts the cleaned reply laid out in lines."""
    cleaned = clean_reply(field.kind, raw)
    return lay_out(cleaned) if field.key in RULE_PROMPTS and isinstance(cleaned, str) else cleaned


# ---------------------------------------------------------------------------
# Checking a value (fixed rules; no judgement)
# ---------------------------------------------------------------------------
def mentions(text: str, word: str, *, any_case: bool = False) -> bool:
    """True if `word` stands in `text` as a whole word (not inside a longer run of letters, digits, "_")."""
    pattern = r"(?<![A-Za-z0-9_])" + re.escape(word) + r"(?![A-Za-z0-9_])"
    return re.search(pattern, text, re.IGNORECASE if any_case else 0) is not None


def check_value(ins: Instructions, key: str, value: str, chosen: dict[str, Value]) -> str | None:
    """Why `value` cannot be the text of the string field `key`, or None if it can.

    `chosen` holds the values of the other fields that exist so far. The rules, each marked as a rule of
    `check_text` (T), a content rule of the design (D), a small rule added here (A), or one of the
    checks added after the three calibrations, after the failed drafting runs of 2026-11-18 and after
    text round 4 (C1 to C21 and the marker lines; the measured problem of each is told at the constants):
      every field       not empty (T); no string of the chat format (T), and no "<|" or "|>", the marks of
                        any other special token that a reply could hold (A); no "```", the rest of a code
                        fence that the cleaning could not remove (A); no "->" if the instruction of the
                        field has notes with that arrow (A); and, as the last rule, the copy rule: no run
                        of `COPY_LIMIT` or more words that stands in the text shown to the writer (A)
      one-line fields   at most 200 characters (D); no colon at the end, which marks an announcement (A)
      several lines     the first line does not end with a colon, which marks an announcement (A); in
                        the two rule prompts it may, if the next line begins with a marker of the layout
      id prefixes       a letter, up to five more letters or digits, then at most one "_" or "-" (A);
                        neither prefix begins with the other (A)
      tool names        an identifier (T); differs from every other tool name (T); does not contain another
                        tool name and is not a part of one (A: the prompt rules below could not be met)
      parameter names   an identifier (T); differs from the name of every other parameter (T inside one tool;
                        A across tools)
      result keys       an identifier (D); differs from every other result key (T); holds no resolution
                        value (T)
      status words      an identifier (A); the two differ (T); holds no resolution value (T)
      resolution values an identifier (A); all differ (T); not inside a result key, a status word, a
                        department name or a role name (T); the value for a request that waits for
                        approval holds a word (cut at the underscores) that begins with a stem of
                        PENDING_STEMS (C11); the value for a role that was held already holds a word
                        that begins with a stem of ALREADY_STEMS, the value for a refused request one
                        of DENIED_STEMS, and no value holds a word that begins with a stem of
                        VAGUE_STEMS (C20)
      confirmations     all differ (T)
      descriptions      a tool's description names no other tool; a parameter's description names no tool but
                        its own (D); the word before "parameter" is the visible name of the parameter (in a
                        tool's description: of a parameter of the tool) or a word of BEFORE_PARAMETER_WORDS
                        (C3); a tool's description holds no visible name of a parameter that only other
                        tools have, unless the word is a result key too (C3)
      error texts       no braces but {value} (T), and {value} at most once; none in the malformed-call
                        text (D); no resolution value as a whole word, in any case: a tool result must
                        not show one (A)
      system prompt     80 to 400 words; names every tool and every resolution value of its rule set; does not
                        hold the extra tool's name, in any case, also not at the start of a longer word (D)
      policy-shift      80 to 400 words; names the tools and the resolution values that its rule set can need;
      prompt            is not equal to the main prompt, also when case and spacing are ignored (D)
      both of these     the program lays out their lines before the checks (`lay_out`); at least
                        MIN_STEP_LINES lines begin with "Step" and a number, and one line for each
                        resolution value of the rule set begins with "Case" and a letter (marker lines);
                        every tool of the rule set is named on a line that names fewer than
                        LISTING_LINE_TOOLS tools of the rule set, the closing tool not counted (C1);
                        every resolution value of the rule set stands on a line with no other value of
                        the rule set (C1); a line that names the people-finding tool, with the line after it,
                        holds "list" and a word that begins with "department" (C2); above the first
                        line with a resolution value of the rule set, one sentence (or two in a row on
                        one line) holds "order", "first" and a word that begins with "case" and names no
                        tool; for each resolution value of the rule set the first line that holds it is
                        taken, and these lines stand in the order of CASE_ORDER (C6); the first line that
                        holds the value meaning REFUSED holds the account word for a switched-off account
                        and not the word for an account in use; the result key of the account state
                        stands on a line that names the people-finding tool or on the line after it, or
                        on that first line together with that tool's name or "step 1" / "step one" (C7)
      both of these,    read as marker lines (`marker_lines`): above the first step line, one sentence
      third calibration tells each of the two working rules (C17); in the line of step 1, a sentence
                        with the result key of the account state or of the manager id begins with a
                        word of FACT_FIRST_WORDS (C12); the line of step 2 holds a word of
                        EVERY_REQUEST_WORDS or a phrase of EVERY_REQUEST_PHRASES (C16); the first lines
                        with the values meaning REFUSED and WAITS FOR APPROVAL hold "person", "step 1"
                        or the people-finding tool (C13); the first line with the value meaning HELD
                        BEFORE holds the result key for a held role, and under the default rule set
                        the first line with the value meaning WAITS FOR APPROVAL holds the result key
                        for a sensitive role (C15); a case that names its tool and then the closing
                        call holds a word of SEQUENCE_WORDS or a phrase of SEQUENCE_PHRASES between
                        them (C14); a sentence that names TOOL_LIST_SIZE or more tools and no
                        resolution value names all five (C18, part iv)
      a prompt of a     (the policy-shift prompt of today) every sentence that names the role-giving tool
      desk that never   holds "not" or "never" and neither "twice" nor "second", and there is such a
      gives a role      sentence; no "case d", no line that begins with "d." or "d)" or "d:", no "four
                        cases" and no "fourth case" (C5); no sentence forbids a second giving; the
                        case with the request for approval, if it speaks of the account, holds the
                        account word for an account in use; the value meaning GIVEN stands only in a
                        sentence with "never", "not" or "no" (C18, parts i to iii)
      a prompt of a     (the policy-shift prompt of today) the result key for a sensitive role and a
      desk where the    word that begins with SENSITIVE_STEM stand in the line of step 2 only, both in
      sensitive flag    any case of letters; the lines that begin with Case carry the marks a, b and c,
      decides nothing   each one time, and each resolution value stands first in the line of its own
                        case; the condition of case c holds only words of LAST_CONDITION_WORDS, the
                        account word for an account in use with the key of the account state, and the
                        key for a held role with "false" or "not" (C21, parts i to v)
      plain prompt     at most a third of the characters of the main prompt; names no resolution value (D);
                        in any case of letters: no status word, no name of one of the six tools, and no
                        word that begins with "sensitiv", "manager" or "approv" (A), or with "giv",
                        "grant" or "assign" (C10); at most PLAIN_MAX_WORDS words (C19); one sentence
                        tells each of the two working rules (C17)
    """
    return _value_problem(ins, key, value, chosen) or copy_problem(ins, key, value)


def _value_problem(ins: Instructions, key: str, value: str, chosen: dict[str, Value]) -> str | None:
    """Every rule of `check_value` but the copy rule."""
    field = ins.fields[key]
    reason = _common_problem(value, one_line=field.kind == "line") or _arrow_problem(field, value)
    if reason is not None:
        return reason
    if field.kind == "text" and _first_line_colon_problem(key, value):
        return "its first line ends with a colon"
    group = key.split(".")[0]
    if group == "id_prefixes":
        return _id_prefix_problem(key, value, chosen)
    if _is_param_name(key):
        return _identifier_problem(value) or _param_name_problem(ins, key, value, chosen)
    if _is_description(key):
        return _description_problem(key, value, chosen)
    if group == "tools":
        return _identifier_problem(value) or _tool_name_problem(key, value, chosen)
    if group == "result_keys" or key.startswith(("values.status.", "values.resolution.")):
        return (_identifier_problem(value) or _unique_problem(ins, key, value, chosen)
                or _resolution_clash(key, value, chosen)
                or (_pending_word_problem(value) if key == "values.resolution.pending" else None)
                or (_resolution_word_problem(key, value) if key in _resolution_keys() else None))
    if key.startswith("values.confirmation."):
        return _unique_problem(ins, key, value, chosen)
    if group == "errors":
        return _error_problem(key, value, chosen)
    if key == "system_prompt":
        return (_prompt_problem(value, chosen, "default") or _extra_tool_problem(value, chosen)
                or _prompt_line_problem(value, chosen, "default")
                or _round3_prompt_problem(value, chosen, "default"))
    if key == "policy_shift.system_prompt":
        ruleset = str(chosen["policy_shift.variant"])
        if normalise(value) == normalise(str(chosen["system_prompt"])):
            return "is equal to the main system prompt"
        return (_prompt_problem(value, chosen, ruleset) or _prompt_line_problem(value, chosen, ruleset)
                or _round3_prompt_problem(value, chosen, ruleset))
    if key == "plain_system_prompt":
        return (_plain_prompt_problem(value, chosen) or _plain_length_problem(value)
                or _working_rules_problem([" ".join(value.split())]))     # as one line: a sentence may wrap
    return None


def _first_line_colon_problem(key: str, value: str) -> bool:
    """True if the first line ends with a colon, which marks an announcement ("Here is the prompt:").

    One exception, for the two rule prompts: the next line that is not empty begins with a marker of the
    line layout. There the colon ends the sentence in front of a step or a case ("Follow these steps in
    order: Step 1 ..."), and the line break behind it may be one that `lay_out` made itself.
    """
    lines = [line for line in value.split("\n") if line.strip()]
    if not value.split("\n", 1)[0].rstrip().endswith(":"):
        return False
    return not (key in RULE_PROMPTS and len(lines) > 1 and _MARKER_LINE.match(lines[1]) is not None)


def check_item(ins: Instructions, key: str, item: str, chosen: dict[str, Value],
               accepted: list[str]) -> str | None:
    """Why `item` cannot be added to the list field `key`, or None if it can. `accepted`: its items so far.

    Rules (T, D, A as at `check_value`):
      every item        not empty (T); no string of the chat format (T) and no "<|" or "|>" (A); at most 200
                        characters (D); no colon at the end (A); no "->" if the instruction has notes with
                        that arrow (A); and, as the last rule, the copy rule of `check_value` (A)
      request template  each of the three variables exactly once and nothing else in braces (T, D); not equal
                        to another template of any style once the variables are removed (D); directly
                        after {role_name} only the end, a punctuation mark or a word of AFTER_ROLE_WORDS
                        (C4); if {department} is the next variable after {role_name}, a mark of
                        ROLE_TO_DEPARTMENT_MARKS or ROLE_TO_DEPARTMENT_WORDS words stand between them,
                        and {role_name} does not stand directly after {department} (C8); no run of
                        SAME_RUN_LIMIT or more words that stands in an accepted template of the field (C9)
      department, role  1 to 4 words, single spaces between them; letters only, hyphens inside a word (D);
                        not equal to an accepted name when case and spacing are ignored (T); holds no
                        resolution value (T)
      role              none of its words is in the instruction file's `sensitivity_words` (D), also not
                        with a plural "s" or "es" at its end, such as "Admins" or "Secrets" (A)
    """
    return _item_problem(ins, key, item, chosen, accepted) or copy_problem(ins, key, item)


def _item_problem(ins: Instructions, key: str, item: str, chosen: dict[str, Value],
                  accepted: list[str]) -> str | None:
    """Every rule of `check_item` but the copy rule."""
    reason = _common_problem(item, one_line=True) or _arrow_problem(ins.fields[key], item)
    if reason is not None:
        return reason
    if key.startswith("request_templates."):
        return _template_problem(key, item, chosen, accepted)
    reason = _name_problem(item, accepted) or _resolution_clash(key, item, chosen)
    if reason is None and key == "roles":
        words = [word.casefold() for word in re.split(r"[ -]", item)]
        hint = next((stem for word in words for stem in (word, word[:-1], word[:-2])
                     if stem in ins.sensitivity_words and word[len(stem):] in ("", "s", "es")), None)
        if hint is not None:
            reason = f"holds the word {hint!r}, which hints at whether a role is sensitive"
    return reason


def _common_problem(value: str, *, one_line: bool) -> str | None:
    if not value.strip():
        return "is empty"
    for marker in FORBIDDEN + SPECIAL_TOKEN_MARKS:
        if marker in value:
            return f"holds {marker!r}, a string of the chat format"
    if FENCE in value:
        return f"holds {FENCE!r}, the rest of a code fence"
    if one_line and len(value) > MAX_LINE_CHARS:
        return f"has {len(value)} characters; one line has at most {MAX_LINE_CHARS}"
    if one_line and value.endswith(":"):
        return "ends with a colon"
    return None


def _arrow_problem(field: Field, value: str) -> str | None:
    """A value may not hold the arrow of the notes, if the instruction of its field has such notes."""
    if NOTE_ARROW in field.instruction and NOTE_ARROW in value:
        return f"holds {NOTE_ARROW!r}, the arrow of the notes in its instruction"
    return None


def _identifier_problem(value: str) -> str | None:
    return None if IDENTIFIER.fullmatch(value) else "is not an identifier (letters, digits, underscores)"


def _id_prefix_problem(key: str, value: str, chosen: dict[str, Value]) -> str | None:
    if not ID_PREFIX.fullmatch(value):
        return "is not a short id prefix (a letter, up to five letters or digits, at most one _ or -)"
    other = chosen.get("id_prefixes.role" if key == "id_prefixes.person" else "id_prefixes.person")
    if isinstance(other, str) and (value.casefold().startswith(other.casefold())
                                   or other.casefold().startswith(value.casefold())):
        return "begins like the other id prefix"
    return None


def _tool_name_problem(key: str, value: str, chosen: dict[str, Value]) -> str | None:
    for tool in ALL_TOOL_KEYS:
        other = chosen.get(tool_name_key(tool))
        if tool_name_key(tool) == key or not isinstance(other, str):
            continue
        if value == other:
            return f"is already the name of another tool ({tool_name_key(tool)})"
        if value.casefold() in other.casefold() or other.casefold() in value.casefold():
            return f"contains the tool name {other!r} or is a part of it"
    return None


def _param_name_problem(ins: Instructions, key: str, value: str, chosen: dict[str, Value]) -> str | None:
    same_parameter = {key, *ins.copies_of(key)}
    for other in ins.fields:
        if _is_param_name(other) and other not in same_parameter and chosen.get(other) == value:
            return f"is already the name of another parameter ({other})"
    return None


def _unique_problem(ins: Instructions, key: str, value: str, chosen: dict[str, Value]) -> str | None:
    """A value must differ from the other values of its group (the fields with the same parent key)."""
    parent = key.rsplit(".", 1)[0]
    for other in ins.fields:
        if other != key and other.rsplit(".", 1)[0] == parent and chosen.get(other) == value:
            return f"is already the value of {other}"
    return None


def _resolution_clash(key: str, value: str, chosen: dict[str, Value]) -> str | None:
    """No resolution value may be readable inside what the read tools return (the rule of check_text)."""
    resolutions = {name: chosen[name] for name in _resolution_keys() if name in chosen}
    if key in _resolution_keys():
        returned = {name: found for name, found in chosen.items()
                    if name.startswith(("result_keys.", "values.status."))}
        for name in ("departments", "roles"):
            returned.update({f"{name}[{i}]": item for i, item in enumerate(chosen.get(name, []))})
        for name, found in returned.items():
            if value.casefold() in str(found).casefold():
                return f"stands inside {name} ({found!r}), which a read tool returns"
        return None
    for found in resolutions.values():
        if str(found).casefold() in value.casefold():
            return f"holds the resolution value {found!r}"
    return None


def _description_problem(key: str, value: str, chosen: dict[str, Value]) -> str | None:
    own = key.split(".")[1]
    for tool in ALL_TOOL_KEYS:
        name = str(chosen[tool_name_key(tool)])
        if tool != own and mentions(value, name, any_case=True):
            return f"names another tool ({name})"
    if len(key.split(".")) == 5:                                    # a parameter's description
        return _parameter_word_problem(value, [str(chosen[key.rsplit(".", 1)[0] + ".name"])])
    names = [str(chosen[name]) for name in _param_name_keys(own)]
    return _parameter_word_problem(value, names) or _foreign_parameter_problem(value, names, chosen)


def _parameter_word_problem(value: str, names: list[str]) -> str | None:
    """Check C3: the word before "parameter" is one of `names` (letter for letter) or no name at all."""
    for word in _BEFORE_PARAMETER.findall(value):
        if word not in names and word.casefold() not in BEFORE_PARAMETER_WORDS:
            return f"calls a parameter {word!r}, which is not its visible name ({', '.join(names)})"
    return None


def _foreign_parameter_problem(value: str, names: list[str], chosen: dict[str, Value]) -> str | None:
    """Check C3, for a tool's description: no visible name of a parameter that only other tools have.

    `names`: the visible parameter names of the tool itself. A word that is a result key too is allowed.
    """
    allowed = set(names) | {str(chosen[f"result_keys.{name}"]) for name in RESULT_KEYS}
    for name in sorted({str(chosen[key]) for key in _param_name_keys()} - allowed):
        if mentions(value, name, any_case=True):
            return f"names the parameter {name!r}, which this tool does not have"
    return None


def format_fields(text: str) -> list[str] | None:
    """The variables of a format string, or None if `str.format` cannot use it or it has a conversion."""
    try:
        parts = list(string.Formatter().parse(text))
    except ValueError:
        return None
    if any(spec or conversion for _, _, spec, conversion in parts):
        return None
    return [name for _, name, _, _ in parts if name is not None]


def _error_problem(key: str, value: str, chosen: dict[str, Value]) -> str | None:
    fields = format_fields(value)
    if fields is None or any(name != "value" for name in fields) or value.count("{") != len(fields) \
            or value.count("}") != len(fields):
        return "holds braces other than {value}"
    if key == "errors.malformed_call" and fields:
        return "holds {value}, but the reply to a malformed call has no variable"
    if len(fields) > 1:
        return "holds {value} more than once"
    for name in _resolution_keys():
        if mentions(value, str(chosen[name]), any_case=True):
            return f"holds the resolution value {chosen[name]!r}, which a tool result must not show"
    return None


def _prompt_problem(value: str, chosen: dict[str, Value], ruleset: str) -> str | None:
    n_words = len(value.split())
    if not PROMPT_WORDS[0] <= n_words <= PROMPT_WORDS[1]:
        return f"has {n_words} words; a system prompt has {PROMPT_WORDS[0]} to {PROMPT_WORDS[1]}"
    for tool in ruleset_tools(ruleset):
        name = str(chosen[tool_name_key(tool)])
        if not mentions(value, name):
            return f"does not name the tool {name}"
    for resolution in ruleset_resolutions(ruleset):
        word = str(chosen[f"values.resolution.{resolution}"])
        if not mentions(value, word):
            return f"does not hold the resolution value {word!r}"
    return None


def _prompt_line_problem(value: str, chosen: dict[str, Value], ruleset: str) -> str | None:
    """The check of the marker lines, then checks C1, C2, C6, C7 and C5, for a prompt that tells the rules
    of `ruleset`. A line: what stands between two line ends, if it is not empty."""
    lines = [line for line in value.split("\n") if line.strip()]
    reason = _marker_line_problem(lines, ruleset) \
        or _tool_line_problem(lines, chosen, ruleset) or _case_line_problem(lines, chosen, ruleset) \
        or _lookup_line_problem(lines, chosen) or _case_order_problem(lines, chosen, ruleset) \
        or _account_source_problem(lines, chosen)
    if reason is None and "grant_role" not in ruleset_tools(ruleset):
        reason = _never_grants_problem(lines, chosen)
    return reason


def _marker_line_problem(lines: list[str], ruleset: str) -> str | None:
    """The first check on lines: enough lines begin with a step marker and with a case marker.

    It stands before the other line checks, so that a prompt in one paragraph is refused for its true
    cause. The program lays the lines out itself (`lay_out`), so this fails only if the sentences of the
    steps and of the cases do not begin with the markers.
    """
    steps = sum(_STEP_LINE.match(line) is not None for line in lines)
    cases = sum(_CASE_LINE.match(line) is not None for line in lines)
    wanted = len(ruleset_resolutions(ruleset))
    if steps >= MIN_STEP_LINES and cases >= wanted:
        return None
    return (f"has too few lines that begin with a marker: {steps} line(s) begin with 'Step' and a number "
            f"(at least {MIN_STEP_LINES} are needed) and {cases} line(s) begin with 'Case' and a letter "
            f"(at least {wanted} are needed); the sentence of each step must begin with the word Step and "
            "its number, and the sentence of each case with the word Case and its letter")


def _tool_line_problem(lines: list[str], chosen: dict[str, Value], ruleset: str) -> str | None:
    """Check C1: every tool of the rule set is named on a line that is not a list of tools.

    Only the tools of the rule set are counted on a line. So the sentence which says that a tool is never
    called (check C5) may stand in the line of a case.
    """
    names = {tool: str(chosen[tool_name_key(tool)]) for tool in ruleset_tools(ruleset)}
    named = [{tool for tool in names if mentions(line, names[tool])} for line in lines]
    counting = [tools for tools in named if len(tools - {"close_request"}) < LISTING_LINE_TOOLS]
    for tool in names:
        if not any(tool in tools for tools in counting):
            return (f"names the tool {names[tool]} only in a list of tools: on every line that holds it, "
                    f"{LISTING_LINE_TOOLS} or more tools of the desk are named besides "
                    f"{names['close_request']}")
    return None


def _case_line_problem(lines: list[str], chosen: dict[str, Value], ruleset: str) -> str | None:
    """Check C1: every resolution value of the rule set stands on a line with no other value of the rule
    set. (A value that the rule set never produces is not counted, as with the tools.)"""
    words = {key: str(chosen[f"values.resolution.{key}"]) for key in ruleset_resolutions(ruleset)}
    held = [{key for key in words if mentions(line, words[key])} for line in lines]
    for key in words:
        if {key} not in held:
            return (f"holds the resolution value {words[key]!r} on no line of its own: every case must be "
                    "a line with one resolution value")
    return None


def _lookup_line_problem(lines: list[str], chosen: dict[str, Value]) -> str | None:
    """Check C2: a line that names the people-finding tool, with the line after it, speaks of the list and
    of the department."""
    name = str(chosen[tool_name_key("lookup_person")])
    stem = re.compile(r"(?<![A-Za-z0-9_])" + LOOKUP_LINE_STEM, re.IGNORECASE)
    for line, after in zip(lines, lines[1:] + [""]):
        both = f"{line}\n{after}"
        if mentions(line, name) and mentions(both, LOOKUP_LINE_WORD, any_case=True) and stem.search(both):
            return None
    return (f"does not say that the person is chosen from the returned list by department: no line that "
            f"names {name}, with the line after it, holds the words {LOOKUP_LINE_WORD!r} and "
            f"{LOOKUP_LINE_STEM!r}")


def _first_line_with(lines: list[str], word: str) -> int | None:
    """The index of the first line that holds `word` as a whole word, or None if no line holds it."""
    return next((n for n, line in enumerate(lines) if mentions(line, word)), None)


def _case_order_problem(lines: list[str], chosen: dict[str, Value], ruleset: str) -> str | None:
    """Check C6: the order of the cases is told before the first case, and the cases stand in that order.

    A resolution value that the rule set never produces is not counted, as in check C1.
    """
    words = [str(chosen[f"values.resolution.{key}"]) for key in CASE_ORDER
             if key in ruleset_resolutions(ruleset)]
    first = [_first_line_with(lines, word) for word in words]
    places = [place for place in first if place is not None]
    names = [str(chosen[tool_name_key(tool)]) for tool in TOOL_KEYS]
    if not any(_tells_order(line, names) for line in lines[:min(places, default=len(lines))]):
        wanted = ", ".join(repr(word) for word in ORDER_LINE_WORDS + (ORDER_LINE_STEM,))
        return ("does not say, before its first case, that the cases are tested in order and that the first "
                "case that fits is followed: above the first line with a resolution value, no sentence "
                f"that names no tool (and no two such sentences in a row on one line) holds the three "
                f"words {wanted}")
    for (earlier, place), (later, after) in zip(zip(words, first), list(zip(words, first))[1:]):
        if place is None or after is None or after <= place:
            return ("its cases do not stand in the order in which they are tested: the first line that "
                    f"holds {later!r} does not stand below the first line that holds {earlier!r}")
    return None


def _tells_order(line: str, names: list[str]) -> bool:
    """Check C6, part 1, for one line: a sentence of it, or two of its sentences that stand one directly
    after the other, hold the words of the order and name none of the tools `names`."""
    stem = re.compile(r"(?<![A-Za-z0-9_])" + ORDER_LINE_STEM, re.IGNORECASE)
    sentences = _SENTENCE_END.split(line)
    for n in range(len(sentences)):
        for text in (sentences[n], " ".join(sentences[n:n + 2])):
            if all(mentions(text, word, any_case=True) for word in ORDER_LINE_WORDS) and stem.search(text) \
                    and not any(mentions(text, name, any_case=True) for name in names):
                return True
    return False


def _account_source_problem(lines: list[str], chosen: dict[str, Value]) -> str | None:
    """Check C7: (i) the case of the switched-off account holds the account word for such an account and
    not the word for an account in use; (ii) the result key of the account state stands where the
    people-finding tool is named, or in that case together with its source (that tool, or step 1)."""
    denied, key = str(chosen["values.resolution.denied"]), str(chosen["result_keys.status"])
    name = str(chosen[tool_name_key("lookup_person")])
    off, in_use = str(chosen["values.status.inactive"]), str(chosen["values.status.active"])
    place = _first_line_with(lines, denied)
    if place is None:
        return f"holds the resolution value {denied!r} on no line"
    case = lines[place]
    where = f"the first line that holds {denied!r} (the case of the switched-off account)"
    if not mentions(case, off, any_case=True):
        return f"{where} does not hold the account word {off!r}"
    if mentions(case, in_use, any_case=True):
        return f"{where} holds the account word {in_use!r}, which is the word for an account in use"
    at_the_tool = any(mentions(line, name) and mentions(f"{line}\n{after}", key)
                      for line, after in zip(lines, lines[1:] + [""]))
    in_the_case = mentions(case, key) and (mentions(case, name) or _STEP_ONE.search(case) is not None)
    if not (at_the_tool or in_the_case):
        return (f"does not show where the account state is read: the result key {key!r} stands neither on "
                f"a line that names {name} (or on the line directly after it), nor on {where} together "
                f"with {name} or step 1")
    return None


def _never_grants_problem(lines: list[str], chosen: dict[str, Value]) -> str | None:
    """Check C5, for the prompt of a rule set under which the role-giving tool is never called."""
    name = str(chosen[tool_name_key("grant_role")])
    naming = [sentence for line in lines for sentence in _SENTENCE_END.split(line)
              if mentions(sentence, name, any_case=True)]
    if not naming:
        return f"does not say that {name} is not called at this desk: no sentence names it"
    if not all(any(mentions(sentence, word, any_case=True) for word in NEGATIONS) for sentence in naming):
        return f"names {name} in a sentence without {' or '.join(repr(word) for word in NEGATIONS)}"
    for word in SECOND_CALL_WORDS:
        if any(mentions(sentence, word, any_case=True) for sentence in naming):
            return f"names {name} in a sentence with {word!r}, which forbids a second call only"
    found = _FOURTH_CASE.search("\n".join(lines))
    if found is not None:
        return f"speaks of a fourth case ({found.group(0).strip()!r}); this desk has three cases: a, b and c"
    return None


# ---------------------------------------------------------------------------
# Checks C12 to C20 (the third calibration) and C21 (text round 4); the problem of each is told at the
# constants
# ---------------------------------------------------------------------------
def marker_lines(value: str) -> list[str]:
    """The lines of a rule prompt as checks C12 to C18 read them: a line runs from one marker to the next.

    A line of the text starts a new line here if it begins with a marker of the layout (Step, Case,
    Finally), or if it is the first line, or if an empty line stands above it. Every other line is joined
    to the line above it, with one space.
    """
    found: list[str] = []
    new = True
    for line in value.split("\n"):
        if not line.strip():
            new = True
        elif new or _MARKER_LINE.match(line) is not None:
            found.append(line.strip())
            new = False
        else:
            found[-1] += " " + line.strip()
    return found


def _round3_prompt_problem(value: str, chosen: dict[str, Value], ruleset: str) -> str | None:
    """Checks C12 to C18 and C21 for a prompt that tells the rules of `ruleset`, in this order: C17, C12,
    C16, C13, C15, C14, C18, C21. They stand behind the older checks, so an older reason is told first."""
    lines = marker_lines(value)
    steps = [n for n, line in enumerate(lines) if _STEP_LINE.match(line) is not None]
    reason = _working_rules_problem(lines[:steps[0]] if steps else lines) \
        or _fact_form_problem(lines, chosen) or _every_request_problem(lines) \
        or _whose_problem(lines, chosen, ruleset) or _case_key_problem(lines, chosen, ruleset) \
        or _two_messages_problem(lines, chosen, ruleset) or _tool_list_problem(lines, chosen)
    if reason is None and "grant_role" not in ruleset_tools(ruleset):
        reason = _second_desk_problem(lines, chosen)
    return reason or _last_case_problem(lines, chosen, ruleset)


def _sentences(lines: list[str]) -> list[str]:
    """Every sentence of `lines`, in their order (a line is cut where `_SENTENCE_END` finds an end)."""
    return [sentence for line in lines for sentence in _SENTENCE_END.split(line) if sentence.strip()]


def _begins(text: str, stems: Iterable[str]) -> bool:
    """True if a word of `text` begins with one of `stems` (any case of letters)."""
    pattern = r"(?<![A-Za-z0-9_])(?:" + "|".join(re.escape(stem) for stem in stems) + ")"
    return re.search(pattern, text, re.IGNORECASE) is not None


def _holds_any(text: str, words: Iterable[str]) -> bool:
    """True if `text` holds one of `words` as a whole word, or one of them that is a phrase with any
    white space between its words (any case of letters)."""
    return any(re.search(r"(?<![A-Za-z0-9_])" + r"\s+".join(re.escape(part) for part in word.split())
                         + r"(?![A-Za-z0-9_])", text, re.IGNORECASE) is not None for word in words)


def _place(text: str, word: str) -> re.Match | None:
    """The first place where `word` stands in `text` as a whole word, letter for letter."""
    return re.search(r"(?<![A-Za-z0-9_])" + re.escape(word) + r"(?![A-Za-z0-9_])", text)


def _step_line(lines: list[str], number: int) -> str | None:
    """The first line that begins with the marker of step `number` ("Step 2", "Step two"), or None."""
    mark = re.compile(rf"\s*Step[ \t]+(?:{number}|(?i:{STEP_NUMBER_WORDS[number - 1]}))(?![A-Za-z0-9_])")
    return next((line for line in lines if mark.match(line) is not None), None)


def _case_line(lines: list[str], chosen: dict[str, Value], resolution: str) -> tuple[str, str | None]:
    """The visible resolution value of `resolution`, and the first line that holds it (None: no line)."""
    word = str(chosen[f"values.resolution.{resolution}"])
    place = _first_line_with(lines, word)
    return word, (None if place is None else lines[place])


def _fact_form_problem(lines: list[str], chosen: dict[str, Value]) -> str | None:
    """Check C12: in the line of step 1, a sentence that holds the result key of the account state or of
    the manager id states a fact: it begins with a word of FACT_FIRST_WORDS."""
    line = _step_line(lines, 1)
    if line is None:
        return "has no line that begins with 'Step 1' (or 'Step one'), which check C12 reads"
    keys = [str(chosen["result_keys.status"]), str(chosen["result_keys.manager_id"])]
    name = str(chosen[tool_name_key("lookup_person")])
    for sentence in _sentences([line]):
        held = [key for key in keys if mentions(sentence, key)]
        head = re.match(r"\s*" + _STEP_MARK + r"[\s:.)\-]*", sentence)
        first = re.search(r"[A-Za-z0-9_]+", sentence[head.end():] if head else sentence)
        word = "" if first is None else first.group(0)
        if held and word.casefold() not in FACT_FIRST_WORDS and word != name:
            return (f"the sentence {sentence.strip()!r} in the line of step 1 holds the result key "
                    f"{held[0]!r} and does not begin with a word that states a fact about the tool result "
                    f"({', '.join(FACT_FIRST_WORDS)}, or {name}): it reads as an order to show or to "
                    "write the value (check C12)")
    return None


def _every_request_problem(lines: list[str]) -> str | None:
    """Check C16: the line of step 2 says that the step is done for every request."""
    line = _step_line(lines, 2)
    if line is None:
        return "has no line that begins with 'Step 2' (or 'Step two'), which check C16 reads"
    if _holds_any(line, EVERY_REQUEST_WORDS + EVERY_REQUEST_PHRASES):
        return None
    wanted = ", ".join(EVERY_REQUEST_WORDS + tuple(f'"{phrase}"' for phrase in EVERY_REQUEST_PHRASES))
    return ("the line of step 2 does not say that the step is done for every request, also when the "
            f"account is switched off: it holds none of {wanted} (check C16)")


def _whose_problem(lines: list[str], chosen: dict[str, Value], ruleset: str) -> str | None:
    """Check C13: the case of the switched-off account and the case with the request for approval say
    whose account state and whose manager is meant."""
    name = str(chosen[tool_name_key("lookup_person")])
    for resolution in ("denied", "pending"):
        word, line = _case_line(lines, chosen, resolution)
        if line is None or resolution not in ruleset_resolutions(ruleset):
            continue
        if not (mentions(line, WHOSE_WORD, any_case=True) or _STEP_ONE.search(line) is not None
                or mentions(line, name)):
            what = "account state" if resolution == "denied" else "manager"
            return (f"the first line that holds {word!r} does not say whose {what} is meant: it holds "
                    f"neither the word {WHOSE_WORD!r}, nor 'step 1', nor {name} (check C13)")
    return None


def _case_key_problem(lines: list[str], chosen: dict[str, Value], ruleset: str) -> str | None:
    """Check C15: the case of the role that is held already holds the result key for "holds the role";
    under the default rule set the case with the request for approval holds the key for "sensitive"."""
    wanted = [("already", "has_access")] + ([("pending", "sensitive")] if ruleset == RULESETS[0] else [])
    for resolution, result_key in wanted:
        word, line = _case_line(lines, chosen, resolution)
        key = str(chosen[f"result_keys.{result_key}"])
        if line is not None and not mentions(line, key):
            return (f"the first line that holds {word!r} does not hold the result key {key!r}, letter for "
                    "letter: the case must name the key that the assistant reads (check C15)")
    return None


def _two_messages_problem(lines: list[str], chosen: dict[str, Value], ruleset: str) -> str | None:
    """Check C14: a case with two calls says that the closing call comes after the other call."""
    close = str(chosen[tool_name_key("close_request")])
    for resolution, tool in (("pending", "request_approval"), ("granted", "grant_role")):
        if resolution not in ruleset_resolutions(ruleset) or tool not in ruleset_tools(ruleset):
            continue
        word, line = _case_line(lines, chosen, resolution)
        name = str(chosen[tool_name_key(tool)])
        first = _place(line, name) if line is not None else None
        if first is None:
            continue                                       # the line names no second call
        last = _place(line, close) or _place(line, word)
        if last.start() < first.start():
            return (f"the first line that holds {word!r} names {close} (or that value) in front of "
                    f"{name}: the closing call must be named after the other call (check C14)")
        end = re.compile(r"[.!?](?:\s|$)").search(line, last.end())
        between = line[first.end():end.start() if end else len(line)]
        if not _holds_any(between, SEQUENCE_WORDS + SEQUENCE_PHRASES):
            wanted = ", ".join(SEQUENCE_WORDS + tuple(f'"{phrase}"' for phrase in SEQUENCE_PHRASES))
            return (f"the first line that holds {word!r} names two calls, {name} and the closing call, "
                    "and does not say that they come one after the other, in two messages: none of "
                    f"{wanted} stands between {name} and the end of the sentence of the closing call "
                    "(check C14)")
    return None


def _working_rules_problem(lines: list[str]) -> str | None:
    """Check C17: `lines` hold the two working rules: one tool call in each message, and an id only from a
    tool result. For a rule prompt `lines` are the lines above the first step line."""
    sentences = _sentences(lines)
    if not any(_holds_any(sentence, ONE_CALL_WORDS) and _begins(sentence, (CALL_STEM,))
               and _begins(sentence, MESSAGE_STEMS) for sentence in sentences):
        return ("does not tell the first working rule (one tool call in each message): no sentence "
                f"{_where(lines)}holds {' or '.join(repr(word) for word in ONE_CALL_WORDS)}, a word "
                f"that begins with {CALL_STEM!r} and a word that begins with one of "
                f"{', '.join(MESSAGE_STEMS)} (check C17)")
    if not any(_holds_any(sentence, ID_WORDS) and _begins(sentence, RESULT_STEMS)
               and _holds_any(sentence, ONLY_WORDS) for sentence in sentences):
        return ("does not tell the second working rule (an id only from a tool result): no sentence "
                f"{_where(lines)}holds {' or '.join(repr(word) for word in ID_WORDS)}, a word that "
                f"begins with {' or '.join(repr(stem) for stem in RESULT_STEMS)} and one of "
                f"{', '.join(ONLY_WORDS)} (check C17)")
    return None


def _where(lines: list[str]) -> str:
    """For the reasons of check C17: the place that was read, as the start of its first line."""
    return f"of the part that begins with {' '.join(lines)[:40]!r} " if lines else ""


def _tool_list_problem(lines: list[str], chosen: dict[str, Value]) -> str | None:
    """Check C18, part iv (both rule prompts): a list of tools in a sentence is the full list of five."""
    names = [str(chosen[tool_name_key(tool)]) for tool in TOOL_KEYS]
    values = [str(chosen[key]) for key in _resolution_keys()]
    for sentence in _sentences(lines):
        named = [name for name in names if mentions(sentence, name)]
        if TOOL_LIST_SIZE <= len(named) < len(names) and not any(mentions(sentence, v) for v in values):
            missing = ", ".join(name for name in names if name not in named)
            return (f"the sentence {sentence.strip()!r} lists {len(named)} tools and leaves out {missing}: "
                    f"a sentence that names {TOOL_LIST_SIZE} or more tools must name all {len(names)}, "
                    "because the assistant has all of them in its list of tools (check C18)")
    return None


def _second_desk_problem(lines: list[str], chosen: dict[str, Value]) -> str | None:
    """Check C18, parts i to iii, for the prompt of a rule set under which no role is given."""
    given = str(chosen["values.resolution.granted"])
    for sentence in _sentences(lines):
        counted = re.sub(r"(?i)second\s+message", " ", sentence)
        if _holds_any(counted, TWICE_WORDS + TWICE_PHRASES) and _begins(counted, ACTION_STEMS):
            return (f"the sentence {sentence.strip()!r} forbids a second giving of a role, and at this "
                    "desk no role is given even once (check C18)")
        if mentions(sentence, given) and not _holds_any(sentence, UNUSED_WORDS):
            return (f"the sentence {sentence.strip()!r} holds the resolution value {given!r}, which this "
                    f"desk never uses, without {', '.join(UNUSED_WORDS[:-1])} or {UNUSED_WORDS[-1]} "
                    "(check C18)")
    word, line = _case_line(lines, chosen, "pending")
    key, in_use = str(chosen["result_keys.status"]), str(chosen["values.status.active"])
    if line is not None and (mentions(line, "account", any_case=True) or mentions(line, key)) \
            and not mentions(line, in_use, any_case=True):
        return (f"the first line that holds {word!r} speaks of the account and does not hold the account "
                f"word {in_use!r}, which the tool result shows for an account in use (check C18)")
    return None


def _last_case_problem(lines: list[str], chosen: dict[str, Value], ruleset: str) -> str | None:
    """Check C21, for the prompt of a rule set that asks for approval and under which no decision depends
    on whether the role is sensitive. Five parts, in this order: (i) only the line of step 2 speaks of a
    sensitive role; (ii) three lines begin with Case, with the marks a, b and c, and each resolution
    value stands first in the line of its own case; (iii) the condition of case c holds only words of the
    list; (iv) no line but the line of step 2 holds a word for a kind of role; (v) no sentence with the
    name of the giving tool says "until", "before" or "after". The line of case c is found as checks C15
    and C18 find it: the first line with the value meaning WAITS FOR APPROVAL."""
    if "pending" not in ruleset_resolutions(ruleset) or _sensitive_decides(ruleset):
        return None
    return (_flag_place_problem(lines, chosen) or _case_marks_problem(lines, chosen, ruleset)
            or _last_condition_problem(lines, chosen) or _role_kind_problem(lines, chosen)
            or _later_giving_problem(lines, chosen))


def _flag_place_problem(lines: list[str], chosen: dict[str, Value]) -> str | None:
    """Check C21, part i: the result key for a sensitive role and a word that begins with SENSITIVE_STEM
    stand in the line of step 2 only. The line of case c is read first, so its reason is told first."""
    key = str(chosen["result_keys.sensitive"])
    word, case = _case_line(lines, chosen, "pending")
    step_2 = _step_line(lines, 2)
    for line in ([] if case is None else [case]) + lines:
        held = None if line is step_2 else _flag_word(line, key)
        if held is None:
            continue
        if line is case:
            return (f"the first line that holds {word!r} holds {held}: at this desk that case is for every "
                    "request that cases a and b did not close, sensitive or not, so its condition is that "
                    "no case before fits, and its line does not speak of a sensitive role (check C21)")
        return (f"the line that begins with {line[:40]!r} holds {held}: at this desk no decision depends "
                "on whether the role is sensitive, so only the line of step 2, which tells what the result "
                "of the role check holds, speaks of it (check C21)")
    return None


def _later_giving_problem(lines: list[str], chosen: dict[str, Value]) -> str | None:
    """Check C21, part v: a sentence that holds the name of the tool that gives a role holds no word of
    LATER_WORDS. At this desk that tool is called at no time, also not after the approval."""
    name = str(chosen[tool_name_key("grant_role")])
    for sentence in _sentences(lines):
        if mentions(sentence, name, any_case=True) and _holds_any(_EVEN_IF.sub(" ", sentence), LATER_WORDS):
            return (f"names {name} in a sentence with one of {', '.join(LATER_WORDS)}: that says the role "
                    "is given later, and at this desk the tool is never called (check C21)")
    return None


def _flag_word(text: str, key: str) -> str | None:
    """What `text` holds of the sensitive flag: the result key `key` or a word with SENSITIVE_STEM."""
    if mentions(text, key, any_case=True):
        return f"the result key {key!r}"
    return f"a word that begins with {SENSITIVE_STEM!r}" if _begins(text, (SENSITIVE_STEM,)) else None


def _role_kind_problem(lines: list[str], chosen: dict[str, Value]) -> str | None:
    """Check C21, part iv: no line but the line of step 2 holds a word that begins with one of
    ROLE_KIND_STEMS. And in the line of step 2, a sentence that holds the result key for a sensitive role,
    the stem or such a word holds no resolution value and no word that begins with one of
    STEP_2_NOT_WITH: that line tells what the result holds, not what to do."""
    key = str(chosen["result_keys.sensitive"])
    step_2 = _step_line(lines, 2)
    for line in lines:
        kind = None if line is step_2 else _KIND.search(line)
        if kind is not None:
            return (f"the line that begins with {line[:40]!r} holds a word for a kind of role "
                    f"({kind.group(0)!r}): at this desk every role is handled in the same way, so the "
                    "prompt speaks of the role with no word for a kind of role (check C21)")
    values = [str(chosen[f"values.resolution.{name}"]) for name in CASE_ORDER]
    for sentence in _sentences([] if step_2 is None else [step_2]):
        kind = _KIND.search(sentence)
        held = _flag_word(sentence, key) or (
            None if kind is None else f"a word for a kind of role ({kind.group(0)!r})")
        other = next((repr(value) for value in values if mentions(sentence, value)), None)
        if other is None and _begins(sentence, STEP_2_NOT_WITH):
            other = "a word that begins with one of " + ", ".join(STEP_2_NOT_WITH)
        if held is not None and other is not None:
            return (f"a sentence of the line of step 2 holds {held} together with {other}: at this desk "
                    "the line of step 2 tells what the result of the role check holds, and no decision "
                    "depends on the kind of the role (check C21)")
    return None


def _case_marks_problem(lines: list[str], chosen: dict[str, Value], ruleset: str) -> str | None:
    """Check C21, part ii: the lines that begin with Case carry the marks a, b and c, each one time; no
    other mark stands behind the word "case"; and the first line that holds a resolution value of the
    rule set begins with the marker of its own case (a case that an empty line cuts in two fails here)."""
    wanted = [key for key in CASE_ORDER if key in ruleset_resolutions(ruleset)]
    marks = list(CASE_MARKS[:len(wanted)])
    told = f"this desk has exactly {len(marks)} lines that begin with Case: {', '.join(marks)}, each one time"
    for line in lines:
        for found in _CASE_NAMED.finditer(line):
            if found.group(1).casefold() not in marks:
                return (f"speaks of a case with the mark {found.group(1)!r} ({found.group(0)!r}); {told} "
                        "(check C21)")
    heads = [found.group(1).casefold() for found in map(_CASE_HEAD.match, lines) if found is not None]
    if sorted(heads) != marks:
        return (f"its lines that begin with Case carry the marks {', '.join(repr(mark) for mark in heads)}; "
                f"{told} (check C21)")
    for mark, resolution in zip(marks, wanted):
        word, line = _case_line(lines, chosen, resolution)
        head = None if line is None else _CASE_HEAD.match(line)
        if line is not None and (head is None or head.group(1).casefold() != mark):
            return (f"the first line that holds {word!r} does not begin with 'Case {mark}': an empty line "
                    "cuts that case in two, or the value stands outside the line of its own case "
                    "(check C21)")
    return None


def last_condition(line: str, chosen: dict[str, Value]) -> str:
    """The condition of a case line, as check C21 reads it: the text between the marker of the line and
    the first of: the word "then", the first name of a tool, the first word of ACTION_WORDS that begins a
    clause, or (with no "then" in the line) the first sentence end after the word "if". With none of
    these: the first sentence of the line."""
    head = _CASE_HEAD.match(line)
    rest = line[head.end():] if head is not None else line
    then, ends = _THEN.search(rest), []
    if then is not None:
        ends.append(then.start())
    names = [str(chosen[tool_name_key(tool)]) for tool in TOOL_KEYS]
    ends += [place.start() for place in (_place(rest, name) for name in names) if place is not None]
    action = _ACTION.search(rest)
    if action is not None:
        ends.append(action.start(1))
    sentence_end = re.compile(r"[.!?](?:\s|$)")
    start = _IF.search(rest)
    if then is None and start is not None and sentence_end.search(rest, start.end()) is not None:
        ends.append(sentence_end.search(rest, start.end()).start())
    if not ends:
        end = sentence_end.search(rest)
        ends.append(len(rest) if end is None else end.start())
    return rest[:min(ends)].strip(" \t:.,;)-")


def _last_condition_problem(lines: list[str], chosen: dict[str, Value]) -> str | None:
    """Check C21, part iii: the condition of case c holds only words of LAST_CONDITION_WORDS, the account
    word for an account in use (and with it the result key of the account state), and the result key for
    "holds the role" together with "false" or "not"."""
    _, line = _case_line(lines, chosen, "pending")
    if line is None:
        return None
    condition = last_condition(line, chosen)
    words = [word.casefold() for word in
             re.findall(r"[A-Za-z0-9_]+", re.sub(r"(?<=\w)['\u2019]s(?![A-Za-z0-9_])", "", condition))]
    state = str(chosen["result_keys.status"]).casefold()
    in_use = str(chosen["values.status.active"]).casefold()
    held = str(chosen["result_keys.has_access"]).casefold()
    allowed = set(LAST_CONDITION_WORDS)
    if in_use in words:
        allowed |= {in_use, state}
    if held in words and ("false" in words or "not" in words):
        allowed |= {held, "false"}
    where = f"the condition of the line of case c ({condition!r})"
    for word in words:
        if word not in allowed:
            return (f"{where} holds the word {word!r}: at this desk that case is for every request that "
                    "cases a and b did not close, so its condition is only that no case above fits; the "
                    "words that the program accepts there are words about the cases and about the account "
                    "in use (check C21)")
    for n, word in enumerate(words):
        if word not in (in_use, "false"):
            continue
        near = words[max(n - 1, 0):n] if n < 2 or words[n - 1] not in ("a", "an", "the", "yet") \
            else words[n - 2:n]
        if near and near[0] in ("not", "no"):
            return (f"{where} holds {' '.join(near + [word])!r}: that turns the condition round, and then a "
                    "request that reaches case c fits no case (check C21)")
    if any(word in FIT_WORDS for word in words) and not any(word in NOT_WORDS for word in words):
        return (f"{where} says that a case fits or that the role is held, and holds none of "
                f"{', '.join(NOT_WORDS)}: case c is for the requests that no case above fits (check C21)")
    if any(word in ROLE_WORDS for word in _ROLE_TALK.sub(" ", " ".join(words)).split()):
        return (f"{where} speaks of the role in another way than \"the person does not hold the role\", "
                "\"the role is not held yet\" or \"for any role\": at this desk case c is for every role "
                "that the person does not hold (check C21)")
    head = _CASE_HEAD.match(line)
    rest = line[(0 if head is None else head.end()):]
    behind = rest.split(condition, 1)[-1] if condition else rest      # no condition: the whole line
    if _holds_any(_EVEN_IF.sub(" ", behind), SECOND_CONDITION_WORDS):
        return (f"the line of case c holds one of {', '.join(SECOND_CONDITION_WORDS)} behind its condition "
                f"({condition!r}): the line has one condition, at its start (check C21)")
    return None


def _plain_length_problem(value: str) -> str | None:
    """Check C19: the plain prompt has at most PLAIN_MAX_WORDS words, counted as `words_of` counts."""
    n_words = len(words_of(value))
    if n_words <= PLAIN_MAX_WORDS:
        return None
    return f"has {n_words} words; the short prompt has at most {PLAIN_MAX_WORDS} (check C19)"


def _resolution_word_problem(key: str, value: str) -> str | None:
    """Check C20: a resolution value tells its own ending and no other one."""
    words = value.casefold().split("_")
    vague = next((word for word in words for stem in VAGUE_STEMS if word.startswith(stem)), None)
    if vague is not None:
        return (f"holds the word {vague!r}, which fits more than one ending of a request: no word of a "
                f"resolution value may begin with one of {', '.join(VAGUE_STEMS)} (check C20)")
    for name, stems, meaning in (("already", ALREADY_STEMS, "the person held the role before"),
                                 ("denied", DENIED_STEMS, "the request was refused")):
        if key == f"values.resolution.{name}" and not any(word.startswith(stem) for word in words
                                                          for stem in stems):
            return (f"does not tell that {meaning}: none of its words (cut at the underscores) begins "
                    f"with one of {', '.join(stems)} (check C20)")
    return None


def _extra_tool_problem(value: str, chosen: dict[str, Value]) -> str | None:
    name = str(chosen[tool_name_key(ALL_TOOL_KEYS[-1])])
    # Also at the start of a longer word ("department_summary_tool"), but not inside one: a short name such
    # as "count" must not refuse every prompt that says "account".
    found = re.search(r"(?<![A-Za-z0-9_])" + re.escape(name), value, re.IGNORECASE)
    return f"names the extra tool ({name})" if found else None


def _plain_prompt_problem(value: str, chosen: dict[str, Value]) -> str | None:
    main = str(chosen["system_prompt"])
    if 3 * len(value) > len(main):
        return f"has {len(value)} characters, more than a third of the main system prompt ({len(main)})"
    for name in _resolution_keys():
        if mentions(value, str(chosen[name]), any_case=True):
            return f"holds the resolution value {chosen[name]!r}"
    for name in _status_keys():
        if mentions(value, str(chosen[name]), any_case=True):
            return f"holds the status word {chosen[name]!r}"
    for tool in ALL_TOOL_KEYS:
        if mentions(value, str(chosen[tool_name_key(tool)]), any_case=True):
            return f"names the tool {chosen[tool_name_key(tool)]}"
    for stem in PLAIN_PROMPT_STEMS:
        if re.search(r"(?<![A-Za-z0-9_])" + re.escape(stem), value, re.IGNORECASE):
            what = ("tells what the desk does with a request (check C10)" if stem in ACTION_STEMS
                    else "belongs to a rule of the desk")
            return f"holds a word that begins with {stem!r}, which {what}"
    return None


def template_skeleton(template: str) -> str:
    """A template without its variables, in the form in which templates are compared."""
    return normalise(_VARIABLE.sub(" ", template))


def _template_problem(key: str, item: str, chosen: dict[str, Value], accepted: list[str]) -> str | None:
    fields = format_fields(item)
    braces_ok = item.count("{") == len(TEMPLATE_VARIABLES) == item.count("}")
    if fields is None or sorted(fields) != sorted(TEMPLATE_VARIABLES) or not braces_ok:
        return "does not hold each of the three variables exactly once and nothing else in braces"
    others = list(accepted)
    for style in STYLES:
        name = f"request_templates.style_{style}"
        others += list(chosen.get(name, [])) if name != key else []
    if template_skeleton(item) in {template_skeleton(other) for other in others}:
        return "is equal to another template when the variables are removed"
    return _after_role_problem(item) or _department_after_role_problem(item) \
        or _role_after_department_problem(item) or _same_run_problem(item, accepted)


def _after_role_problem(item: str) -> str | None:
    """Check C4: what stands directly after {role_name} cannot be read as a part of the role's name."""
    after = _AFTER_ROLE.search(item).group(1)
    if after == "{":
        return "another variable stands directly after {role_name}"
    if after is not None and after.casefold() not in AFTER_ROLE_WORDS:
        return (f"the word {after!r} stands directly after {{role_name}}, where a reader could take it for "
                "a part of the role's name")
    return None


def _department_after_role_problem(item: str) -> str | None:
    """Check C8: if {department} is the next variable after {role_name}, a mark of
    ROLE_TO_DEPARTMENT_MARKS or at least ROLE_TO_DEPARTMENT_WORDS words stand between the two."""
    start, end = item.index("{role_name}") + len("{role_name}"), item.index("{department}")
    if end < start or "{person_name}" in item[start:end]:
        return None                                    # {department} is not the next variable
    between = item[start:end]
    words = re.findall(r"\w+", _INSIDE_WORD_MARK.sub("", between))
    if any(mark in between for mark in ROLE_TO_DEPARTMENT_MARKS) or len(words) >= ROLE_TO_DEPARTMENT_WORDS:
        return None
    marks = " ".join(ROLE_TO_DEPARTMENT_MARKS)
    return (f"only {len(words)} word(s) ({' '.join(words)!r}) and none of the marks {marks} stand between "
            "{role_name} and {department}, so a reader could take the department for a part of the role's "
            f"name; at least {ROLE_TO_DEPARTMENT_WORDS} words or one of these marks must stand there")


def _role_after_department_problem(item: str) -> str | None:
    """Check C8, the mirrored order: {role_name} does not stand directly after {department}."""
    if _DEPARTMENT_THEN_ROLE.search(item) is None:
        return None
    return ("{role_name} stands directly after {department}, with nothing but white space between them, "
            "so a reader could take the department for the first part of the role's name")


def _pending_word_problem(value: str) -> str | None:
    """Check C11: the value for a request that waits for approval tells that meaning."""
    if any(word.startswith(stem) for word in value.casefold().split("_") for stem in PENDING_STEMS):
        return None
    return ("does not tell that the request waits for approval: none of its words (cut at the underscores) "
            f"begins with one of {', '.join(PENDING_STEMS)} (check C11)")


def _same_run_problem(item: str, accepted: list[str]) -> str | None:
    """Check C9: no run of SAME_RUN_LIMIT or more words that stands in a template of the same field which
    was accepted before this one."""
    run = longest_shared_run(words_of(item), [words_of(other) for other in accepted])
    if len(run) < SAME_RUN_LIMIT:
        return None
    return (f"is nearly the same sentence as a template that was accepted before it: the two share a run "
            f"of {len(run)} words ({' '.join(run)!r}); the limit is {SAME_RUN_LIMIT}")


def _name_problem(item: str, accepted: list[str]) -> str | None:
    words = item.split(" ")
    well_formed = all(word[:1].isalpha() and word[-1:].isalpha()
                      and all(ch.isalpha() or ch == "-" for ch in word) for word in words)
    if not well_formed or not NAME_WORDS[0] <= len(words) <= NAME_WORDS[1]:
        return f"is not {NAME_WORDS[0]} to {NAME_WORDS[1]} words of letters and hyphens"
    if normalise(item) in {normalise(other) for other in accepted}:
        return "is in the list already"
    return None


# ---------------------------------------------------------------------------
# Asking the writer and taking the first valid candidate
# ---------------------------------------------------------------------------
@dataclass
class DraftResult:
    chosen: dict[str, Value]        # the value of every field that has one
    failed: dict[str, str]          # field -> why it has no value

    @property
    def complete(self) -> bool:
        return not self.failed


def draft_all(ins: Instructions, writer: Writer, *, writer_id: str, run_seed: int, n_candidates: int = 8,
              only: Iterable[str] | None = None, previous: dict[str, Value] | None = None,
              cache: dict[str, dict] | None = None,
              emit: Callable[[dict], None] | None = None) -> DraftResult:
    """Write every field, in the order of the instruction file. Each step is given to `emit` as a log line.

    `writer_id` names the writer in the log (model id and revision).
    `only`: write only these fields and every field that depends on one of them; the other fields keep their
        value from `previous` (the values of an earlier run, see `previous_values`). Fixed fields always take
        their value from the instruction file.
    `cache`: replies of an earlier run (see `reply_cache`). A candidate with the same writer, seed, token
        limit and messages is not asked for again; its line is marked `"reused": true`.
    A field with no valid candidate is in `failed`, and so is every field that needs its value.
    """
    emit = emit if emit is not None else (lambda line: None)
    todo = fields_to_write(ins, only)
    kept = kept_values(ins, only, previous)
    result = DraftResult(chosen=dict(kept), failed={})
    for key in ins.order():
        field = ins.fields[key]
        absent = [name for name in field.depends_on() if name not in result.chosen]
        if field.kind == "fixed":
            result.chosen[key] = field.value
            emit({"kind": "fixed", "field": key, "value": field.value, "source": field.source})
        elif key not in todo:
            emit({"kind": "kept", "field": key, "value": kept[key]})
        elif absent:
            result.failed[key] = f"not written, because these fields have no value: {absent}"
            emit({"kind": "failed", "field": key, "reason": result.failed[key]})
        elif field.kind == "copy":
            result.chosen[key] = result.chosen[field.source]
            emit({"kind": "copied", "field": key, "value": result.chosen[key], "from": field.source})
        else:
            ask = _Asker(writer, writer_id, run_seed, cache or {})
            value = _draft_field(ins, field, result.chosen, ask, n_candidates, emit)
            if value is None:
                result.failed[key] = f"no valid candidate among {n_candidates}"
                emit({"kind": "failed", "field": key, "reason": result.failed[key]})
            else:
                result.chosen[key] = value
    return result


def fields_to_write(ins: Instructions, only: Iterable[str] | None) -> list[str]:
    """Every field (`only` is None), or the named fields and every field that depends on one of them."""
    if only is None:
        return ins.order()
    named = list(only)
    unknown = [key for key in named if key not in ins.fields]
    if unknown:
        raise DraftError(f"unknown field(s) {unknown}; the fields are listed by --dry-run")
    todo: list[str] = []
    for key, field in ins.fields.items():
        if key in named or any(name in todo for name in field.depends_on()):
            todo.append(key)
    return todo


def kept_values(ins: Instructions, only: Iterable[str] | None,
                previous: dict[str, Value] | None) -> dict[str, Value]:
    """The values of the fields that a run with `only` does not write again, taken from `previous`.

    Raises DraftError if `previous` lacks one of them. Fixed fields are not in the result: they always take
    their value from the instruction file.
    """
    todo = fields_to_write(ins, only)
    kept = {key: (previous or {}).get(key) for key, field in ins.fields.items()
            if key not in todo and field.kind != "fixed"}
    missing = [key for key, value in kept.items() if not _is_complete(ins.fields[key], value)]
    if missing:
        raise DraftError(f"the log holds no value for {missing}: write these fields too, or run without "
                         "--only")
    return kept


def _is_complete(field: Field, value: Value | None) -> bool:
    if field.kind == "list":
        return isinstance(value, list) and len(value) >= field.want
    return isinstance(value, str)


class _Asker:
    """Gets the raw reply of one candidate: from the cache if it is there, else from the writer."""

    def __init__(self, writer: Writer, writer_id: str, run_seed: int, cache: dict[str, dict]) -> None:
        self.writer, self.writer_id, self.run_seed, self.cache = writer, writer_id, run_seed, cache

    def __call__(self, field: Field, index: int, messages: list[dict]) -> dict:
        """The start of a candidate's log line: who was asked what, and the raw reply."""
        seed = candidate_seed(self.run_seed, field.key, index)
        limit = MAX_NEW_TOKENS[field.kind]
        line = {"kind": "candidate", "field": field.key, "index": index, "seed": seed,
                "writer": self.writer_id, "max_new_tokens": limit, "messages": messages}
        found = self.cache.get(cache_key(line))
        if found is not None:
            return {**line, "raw": found["raw"], "cut_off": found["cut_off"], "reused": True}
        try:
            raw, cut_off = self.writer(messages, seed, limit), False
        except ReplyCutOff as stopped:
            raw, cut_off = stopped.text, True
        return {**line, "raw": raw, "cut_off": cut_off, "reused": False}


def cache_key(line: dict) -> str:
    """What makes a reply reusable: the same writer, seed, token limit and messages."""
    return json.dumps([line["writer"], line["seed"], line["max_new_tokens"], line["messages"]],
                      ensure_ascii=False, sort_keys=True)


def _draft_field(ins: Instructions, field: Field, chosen: dict[str, Value], ask: _Asker, n_candidates: int,
                 emit: Callable[[dict], None]) -> Value | None:
    """Ask for candidates of one field until it is filled. Returns its value, or None if it stays unfilled.

    A string field takes the first valid candidate. A list field takes valid new items, reply after reply,
    until it has `field.want` of them. No candidate is asked for once the field is filled.
    The log line of every candidate, valid or not, holds `shared_run`: the longest run of words that it
    shares with the text shown to the writer (for a list: the longest among its items).
    """
    messages = build_messages(ins, field.key, chosen)
    accepted: list[str] = []
    for index in range(n_candidates):
        line = ask(field, index, messages)
        cleaned = cleaned_value(field, line["raw"])
        if field.kind == "list":
            line.update(_judge_items(ins, field, cleaned, chosen, accepted, line["cut_off"]))
            accepted += [entry["value"] for entry in line["items"] if entry["taken"]]
            done = len(accepted) == field.want
        else:
            reason = CUT_OFF_REASON if line["cut_off"] else check_value(ins, field.key, cleaned, chosen)
            valid = reason is None
            line.update({"cleaned": cleaned, "valid": valid, "reason": reason, "taken": valid,
                         "shared_run": run_entry(shared_run(ins, field.key, cleaned))})
            done = reason is None
        emit(line)
        if done:
            return accepted if field.kind == "list" else cleaned
    return None


def _judge_items(ins: Instructions, field: Field, items: list[str], chosen: dict[str, Value],
                 accepted: list[str], cut_off: bool) -> dict:
    """The second half of a list candidate's log line: each item with valid or not, why, and taken or not.

    Valid items are taken in the order of the reply until the field has `field.want` items. A reply that was
    cut off by the token limit gives no item, because its last line may be incomplete.
    """
    entries: list[dict] = []
    taken: list[str] = []
    for item in items:
        reason = CUT_OFF_REASON if cut_off else check_item(ins, field.key, item, chosen, accepted + taken)
        take = reason is None and len(accepted) + len(taken) < field.want
        entries.append({"value": item, "valid": reason is None, "reason": reason, "taken": take,
                        "shared_run": run_entry(shared_run(ins, field.key, item))})
        taken += [item] if take else []
    reason = None if taken else (CUT_OFF_REASON if cut_off else "the reply holds no valid new item")
    longest = max((entry["shared_run"] for entry in entries), key=lambda run: run["length"],
                  default=run_entry([]))
    return {"cleaned": items, "items": entries, "valid": bool(taken), "reason": reason, "taken": bool(taken),
            "shared_run": longest}


# ---------------------------------------------------------------------------
# Reading a log again (--resume and --only)
# ---------------------------------------------------------------------------
def read_log(path: str | Path) -> list[dict]:
    """The lines of a drafting log; an empty list if the file does not exist."""
    if not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def reply_cache(lines: list[dict]) -> dict[str, dict]:
    """The raw replies in a log, by `cache_key`."""
    return {cache_key(line): {"raw": line["raw"], "cut_off": line["cut_off"]}
            for line in lines if line.get("kind") == "candidate"}


def _last_taken(lines: list[dict]) -> dict[str, list[tuple[dict, dict]]]:
    """For each field that a log asked for: the candidate lines that were taken in the LAST run that asked
    for it, each with the "run" line of that run. An empty list: that run found no valid candidate.

    A log may hold several runs, each started by a line of kind "run".
    """
    found: dict[str, list[tuple[dict, dict]]] = {}
    run: dict = {}
    asked_in_run: set[str] = set()
    for line in lines:
        kind, key = line.get("kind"), line.get("field")
        if kind == "run":
            run, asked_in_run = line, set()
        elif kind == "candidate":
            if key not in asked_in_run:
                found[key] = []
                asked_in_run.add(key)
            if line["taken"]:
                found[key].append((run, line))
    return found


def previous_values(lines: list[dict]) -> dict[str, Value]:
    """The value that each field has after the last run in a log.

    A run that writes a field again replaces the field's earlier value; if that run found no valid
    candidate, the field has no value.
    """
    values: dict[str, Value] = {line["field"]: line["value"] for line in lines
                                if line.get("kind") in ("fixed", "copied")}
    for key, taken in _last_taken(lines).items():
        if taken and "items" in taken[0][1]:
            values[key] = [entry["value"] for _, line in taken for entry in line["items"] if entry["taken"]]
        elif taken:
            values[key] = taken[0][1]["cleaned"]
    return values


def kept_sources(lines: list[dict], kept: Iterable[str]) -> dict[str, list]:
    """Where the values of the fields in `kept` come from, read from a log: the writers, and the run seeds
    and instruction-file hashes of the runs that wrote them. Used when a run with `only` keeps them."""
    taken = _last_taken(lines)
    pairs = [pair for key in kept for pair in taken.get(key, [])]
    runs = [run for run, _ in pairs]
    return {"writers": sorted({str(line.get("writer")) for _, line in pairs}),
            "run_seeds": sorted({run["run_seed"] for run in runs if "run_seed" in run}),
            "instructions_sha256": sorted({run["instructions_sha256"] for run in runs
                                           if "instructions_sha256" in run})}


def kept_problems(ins: Instructions, chosen: dict[str, Value], kept: Iterable[str]) -> dict[str, str]:
    """For each kept value that today's checks would refuse: why. `chosen` holds every field of the text.

    A run with `only` does not write the kept values again and does not check them: they passed the checks
    of the run that wrote them. The checks or the instruction file may have changed since then, so this
    measures the kept values (those that a model wrote) against `check_value` and `check_item` as they are
    now. It is a report for the header of the text file; it refuses nothing.
    """
    found: dict[str, str] = {}
    for key in kept:
        if not ins.fields[key].asked:
            continue
        others = {name: value for name, value in chosen.items() if name != key}
        if ins.fields[key].kind == "list":
            items = list(chosen[key])
            reasons = [f"{item!r} {reason}" for n, item in enumerate(items)
                       if (reason := check_item(ins, key, item, others, items[:n])) is not None]
        else:
            reason = check_value(ins, key, str(chosen[key]), others)
            reasons = [reason] if reason is not None else []
        if reasons:
            found[key] = "; ".join(reasons)
    return found


# ---------------------------------------------------------------------------
# What a log says about copying and about drawing again (for the header of the text file)
# ---------------------------------------------------------------------------
def taken_runs(lines: list[dict]) -> dict[str, dict | None]:
    """For each field that has a taken value after the last run of a log: its `shared_run` from the log.

    For a list field it is the longest among the taken items. None: a line of the field holds no
    measurement (it was written before the copy rule existed).
    """
    found: dict[str, dict | None] = {}
    for key, taken in _last_taken(lines).items():
        runs = [entry.get("shared_run") for _, line in taken
                for entry in ([e for e in line["items"] if e["taken"]] if "items" in line else [line])]
        if runs:
            found[key] = None if None in runs else max(runs, key=lambda run: run["length"])
    return found


def draw_counts(lines: list[dict]) -> dict[str, int]:
    """For each field that a log asked for: in how many runs of the log a new reply was drawn for it.

    A run is what follows one line of kind "run". A reply that `--resume` took from the log is not a new
    draw. A field that was drawn in more than one run was written again; the count makes that visible.
    """
    counts: dict[str, int] = {}
    drawn_in_run: set[str] = set()
    for line in lines:
        key = line.get("field")
        if line.get("kind") == "run":
            drawn_in_run = set()
        elif line.get("kind") == "candidate" and not line.get("reused") and key not in drawn_in_run:
            drawn_in_run.add(key)
            counts[key] = counts.get(key, 0) + 1
    return counts


def log_report(ins: Instructions, lines: list[dict]) -> dict:
    """What the header of the text file says about a log (all the runs in it, oldest first).

    "largest_run": the largest shared run among the values that were taken, with its field
                   ({"field", "length", "words"}), or None if the log holds no measured value.
    "unmeasured":  the fields whose taken value has no measurement in the log.
    "draws":       for every field that is asked from the model, in the order of the instruction file, in
                   how many runs a value for it was drawn (`draw_counts`).
    """
    runs = taken_runs(lines)
    measured = [{"field": key, **run} for key, run in runs.items() if run is not None]
    counts = draw_counts(lines)
    return {"largest_run": max(measured, key=lambda run: run["length"], default=None),
            "unmeasured": [key for key, run in runs.items() if run is None],
            "draws": {key: counts.get(key, 0) for key, field in ins.fields.items() if field.asked}}


def _leaf_strings(node: object) -> list[str]:
    """Every string value in nested mappings and lists, in their order."""
    if isinstance(node, dict):
        return [text for value in node.values() for text in _leaf_strings(value)]
    if isinstance(node, list):
        return [text for value in node for text in _leaf_strings(value)]
    return [node] if isinstance(node, str) else []


def placeholder_report(ins: Instructions, chosen: dict[str, Value],
                       path: str | Path | None = None) -> dict | None:
    """The longest run of words that a value written by the model shares with a value of the placeholder
    text file (`text/placeholder_text.yaml` unless `path` names another file).

    The writer never sees that file, so a long run could only have come through the instructions. This is a
    report and not a rule: nothing is refused because of it. Short values share a word or two by nature (a
    key such as "role_id", a department such as "Finance"). Returns {"field", "length", "words"}; the field
    is None if no word is shared. Returns None if the file does not exist.
    """
    path = Path(path) if path is not None else placeholder_path()
    if not path.exists():
        return None
    texts = [words_of(text) for text in _leaf_strings(yaml.safe_load(path.read_text(encoding="utf-8")))]
    best: dict = {"field": None, **run_entry([])}
    for key, field in ins.fields.items():
        if field.asked and key in chosen:
            value = chosen[key]
            for item in value if isinstance(value, list) else [value]:
                run = longest_shared_run(words_of(item), texts)
                if len(run) > best["length"]:
                    best = {"field": key, **run_entry(run)}
    return best


def shown_run(run: dict | None) -> str:
    """A reported run as one piece of a sentence: `6 words in errors.x ("no person has the id {value}")`."""
    if run is None:
        return "not measured"
    if not run["length"]:
        return "0 words"
    return f"{run['length']} word{'s' if run['length'] != 1 else ''} in {run['field']} (\"{run['words']}\")"


# ---------------------------------------------------------------------------
# Writing the text file
# ---------------------------------------------------------------------------
def assemble(chosen: dict[str, Value], written_by: str) -> dict:
    """The content of the text file as nested mappings, in the file's order. Every field must be there."""
    data: dict = {"meta": {"written_by": written_by, "placeholder": False}}
    strings, lists = text_fields()
    for key in strings + lists:
        node = data
        *parents, last = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[last] = list(chosen[key]) if key in lists else chosen[key]
    return data


class _Dumper(yaml.SafeDumper):
    """SafeDumper that writes a string of several lines as a block (`|`), so that a prompt stays readable."""


def _represent_str(dumper: yaml.SafeDumper, value: str) -> yaml.Node:
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="|" if "\n" in value else None)


_Dumper.add_representer(str, _represent_str)


def render_yaml(data: dict) -> str:
    """`data` as YAML: the top-level keys in their order, one empty line between them, no line folding."""
    blocks = [yaml.dump({key: value}, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=10**6,
                        default_flow_style=False) for key, value in data.items()]
    return "\n".join(blocks)


def header_comment(ins: Instructions, *, model_id: str, revision: str, written_at_utc: str, run_seed: int,
                   n_candidates: int, sampling: str, instructions_file: str, log_file: str, report: dict,
                   kept: dict[str, list] | None = None, written: list[str] | None = None,
                   refused_now: dict[str, str] | None = None) -> str:
    """The comment at the top of the written text file: who wrote what, when, from which instructions.

    `report`: the result of `log_report` for the whole log. The comment gives the largest shared run among
    the taken values (one line), and for each field in how many runs a value for it was drawn.
    `kept`: the result of `kept_sources`, for a run that wrote only some fields again. The comment then says
    that the other values come from earlier runs, with their run seeds and instruction-file hashes. If one
    of those hashes is not the hash of `ins`, it says in plain words that the instruction file was changed
    in between, and under which file which values were written.
    `written`: with `kept`, the fields that this run wrote (asked from the model). They are listed.
    `refused_now`: with `kept`, the result of `kept_problems`. The comment says whether every kept value
    passes the checks as they are now, and lists those that do not.
    """
    names = ", ".join(ins.fields[tool_name_key(tool)].value for tool in TOOL_KEYS)
    lines = [
        "The text that the model reads. An open-weight model wrote the values in this file, one field at a",
        "time. The values that it did not write are named below (tool names, repeated parameter names).",
        "",
        f"Written by: {model_id}, revision {revision}",
        f"Date and time (UTC): {written_at_utc}",
        f"Run seed: {run_seed}",
        f"Sampling of the writing model: {sampling}",
        "Instructions: written by the coding assistant (a closed API model). They are in",
        f"  {instructions_file} (SHA-256 {ins.sha256}).",
        f"Selection: up to {n_candidates} candidates per field; each reply was cleaned and checked by fixed",
        "  rules (accessdesk/drafting.py), and the first valid candidate in index order was taken. Nobody",
        f"  chose between candidates: no person and no closed model. Every candidate is in {log_file}.",
        "Coding assistant: a coding assistant (a closed API model) wrote the instruction file and the",
        "  checks, read each drafted text, and ordered new draws of the fields that it judged wrong. It",
        "  wrote and edited no value and chose no candidate. In each run the program took the first",
        "  candidate that passed the checks.",
        "Line layout: in the two system prompts that tell the rules, the program starts a new line in front",
        "  of each sentence that begins with the word Step and a number, with the word Case and a letter,",
        "  or with the word Finally. This changes no word: only white space becomes a line break.",
        f"Tool names: the names of the five tools follow the brief's example list ({names}).",
        "  The writing model was not asked for them. A parameter that means the same thing in several tools",
        "  was asked for once, and the program put that name into each of those tools.",
        "Person names: they are not in this file. They come from the public lists in data/names/.",
        "Keys on the left are internal and are never shown to the model.",
    ]
    if kept is not None:
        lines += [
            "Kept values: the run named above wrote only some fields again (--only). The other values were",
            f"  kept from earlier runs of the same model in the log. Their run seeds: {kept['run_seeds']}.",
            "  SHA-256 of their instruction files:",
            *[f"    {digest}" for digest in kept["instructions_sha256"]],
        ]
        if any(digest != ins.sha256 for digest in kept["instructions_sha256"]):
            lines += [
                "  The instruction file was changed between runs: a hash in this list differs from the",
                f"  hash at 'Instructions' above ({ins.sha256}).",
                "  The kept values were written under the files of this list. Only the fields that the run",
                "  named above wrote were written under the file at 'Instructions'. What the writer was",
                f"  shown for each kept value stands in the 'messages' of its line in {log_file}.",
            ]
        if written is not None:
            lines += ["  Fields that the run named above wrote (every other value that the model wrote is a",
                      "  kept value):",
                      *[f"    {key}" for key in written]]
        if refused_now is not None:
            verdict = ("every kept value passes them." if not refused_now
                       else f"{len(refused_now)} kept value(s) would be refused:")
            lines += ["  Kept values and the checks as they are now (the kept values were not checked again",
                      f"  when this file was written; this line is a report): {verdict}",
                      *[f"    {key}: {reason}" for key, reason in refused_now.items()]]
    lines += [
        f"Copy check: a candidate that shared a run of {COPY_LIMIT} or more words with the text shown to its",
        "  writer (the instructions) was refused. The limit is our judgement (accessdesk/drafting.py).",
        f"Largest shared run among the values that were taken: {shown_run(report['largest_run'])}",
    ]
    if report["unmeasured"]:
        lines += [f"  Not measured (their log lines are older than the copy check): {report['unmeasured']}"]
    lines += [
        f"Runs in which a value was drawn, for each field (counted in {log_file}; a field with more than 1",
        "  was written again):",
        *[f"  {key}: {count}" for key, count in report["draws"].items()],
    ]
    lines += ["", "Read and approved by: (not yet)"]
    return "".join(f"# {line}".rstrip() + "\n" for line in lines)


def write_text_file(out_path: str | Path, content: str) -> list[str]:
    """Write `content` to `out_path` only if `check_text` accepts it. Returns the problems (empty = written).

    The content is first written beside the target, under the target's name plus ".checking", and checked
    there. That file is removed again if the check fails, and `out_path` is not touched.
    """
    out_path = Path(out_path)
    checking = out_path.with_name(out_path.name + ".checking")
    checking.write_text(content, encoding="utf-8")
    problems = check_text(checking)
    if problems:
        checking.unlink()
        return problems
    os.replace(checking, out_path)
    return []


# ---------------------------------------------------------------------------
# Reading every prompt without a model (--dry-run)
# ---------------------------------------------------------------------------
def standin_values(ins: Instructions) -> dict[str, Value]:
    """A made-up value for every field, so that every prompt can be shown: `<<field key>>`.

    Fixed fields have their real value, and a copy has the stand-in of the field it copies.
    """
    values: dict[str, Value] = {}
    for key, field in ins.fields.items():
        if field.kind == "fixed":
            values[key] = field.value
        elif field.kind == "copy":
            values[key] = values[field.source]
        elif field.kind == "list":
            values[key] = [f"<<{key}, item {n}>>" for n in (1, 2)]
        else:
            values[key] = f"<<{key}>>"
    return values


def dry_run_text(ins: Instructions, run_seed: int) -> str:
    """Every field in the order of writing, with the exact messages that the writer would get.

    Values of earlier fields are stand-ins (`standin_values`), because no model has written them.
    """
    values = standin_values(ins)
    blocks: list[str] = []
    for number, (key, field) in enumerate(ins.fields.items(), start=1):
        head = f"=== {number}/{len(ins.fields)}  {key}  "
        if field.kind == "fixed":
            blocks.append(f"{head}(fixed, not asked) ===\nvalue: {field.value}\nsource: {field.source}")
        elif field.kind == "copy":
            blocks.append(f"{head}(copied by the program, not asked) ===\nsame value as: {field.source}")
        else:
            system, user = build_messages(ins, key, values)
            kept = f"; the first {field.want} valid items are kept" if field.kind == "list" else ""
            head += (f"(ask: {field.kind}{kept}; at most {MAX_NEW_TOKENS[field.kind]} new tokens; "
                     f"seed of candidate 0: {candidate_seed(run_seed, key, 0)}) ===")
            blocks.append(f"{head}\n--- system message ---\n{system['content']}\n--- user message ---\n"
                          f"{user['content']}")
    return "\n\n".join(blocks) + "\n"
