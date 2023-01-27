import streamlit as st
from IPython.display import display, HTML
from tkinter import font
import streamlit.components.v1 as components
from semgrex_visualizer import visualize_search_str
from semgrex_visualizer import edit_html_overflow
import os

os.environ['CLASSPATH'] = "C:\\stanford-corenlp-4.5.2\\stanford-corenlp-4.5.2\\*"

st.title("Displaying Semgrex Queries")

SAMPLE_HTML = '''
<!DOCTYPE html>
<html>


<body>

<div class={min-height:100%}>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:lang="en" id="9f9bcae06920463e98e7992840fd8b54-0" class="displacy" width="850" height="402" direction="ltr" style="max-width: none; height: 402px; color: #000000; background: #ffffff; font-family: Arial; direction: ltr; overflow: visible; display: block">
    <style> .bolded{font-weight: bold;} </style>
    <text class="displacy-token" fill="currentColor" text-anchor="middle" y="182.0">
        <tspan class="bolded" fill="#66CCEE" x="50">Banning</tspan>

       <tspan class="displacy-tag" dy="2em" fill="currentColor" x="50">VERB</tspan>
      <tspan class="displacy-word" dy="2em" fill="#66CCEE" x=50>Act.</tspan>
    </text>

    <text class="displacy-token" fill="currentColor" text-anchor="middle" y="182.0">
        <tspan class="bolded" fill="#66CCEE" x="150">opal</tspan>

       <tspan class="displacy-tag" dy="2em" fill="currentColor" x="150">NOUN</tspan>
      <tspan class="displacy-word" dy="2em" fill="#66CCEE" x=150>Thi.</tspan>
    </text>

    <text class="displacy-token" fill="currentColor" text-anchor="middle" y="182.0">
        <tspan class="bolded" fill="#4477AA" x="250">removed</tspan>

       <tspan class="displacy-tag" dy="2em" fill="currentColor" x="250">VERB</tspan>
      <tspan class="displacy-word" dy="2em" fill="#4477AA" x=250>Act.</tspan>
      <tspan class="displacy-word" dy="2em" fill="#4477AA" x=250>Act.</tspan>
    </text>

    <text class="displacy-token" fill="currentColor" text-anchor="middle" y="182.0">
        <tspan class="displacy-word" fill="currentColor" x="350">artifact</tspan>
        <tspan class="displacy-tag" dy="2em" fill="currentColor" x="350">NOUN</tspan>
    </text>

    <text class="displacy-token" fill="currentColor" text-anchor="middle" y="182.0">
        <tspan class="bolded" fill="#4477AA" x="450">decks</tspan>

       <tspan class="displacy-tag" dy="2em" fill="currentColor" x="450">NOUN</tspan>
      <tspan class="displacy-word" dy="4em" fill="#4477AA" x=450>Thi.</tspan>
    </text>

    <text class="displacy-token" fill="currentColor" text-anchor="middle" y="182.0">
        <tspan class="displacy-word" fill="currentColor" x="550">from</tspan>
        <tspan class="displacy-tag" dy="2em" fill="currentColor" x="550">ADP</tspan>
    </text>

    <text class="displacy-token" fill="currentColor" text-anchor="middle" y="182.0">
        <tspan class="displacy-word" fill="currentColor" x="650">the</tspan>
        <tspan class="displacy-tag" dy="2em" fill="currentColor" x="650">DET</tspan>
    </text>

    <text class="displacy-token" fill="currentColor" text-anchor="middle" y="182.0">
        <tspan class="bolded" fill="#4477AA" x="750">meta .</tspan>

       <tspan class="displacy-tag" dy="2em" fill="currentColor" x="750">NOUN</tspan>
      <tspan class="displacy-word" dy="2em" fill="#4477AA" x=750>Obj.</tspan>
    </text>

    <g class="displacy-arrow">
        <path class="displacy-arc" id="arrow-9f9bcae06920463e98e7992840fd8b54-0-0" stroke-width="2px" d="M50,152.0 50,118.66666666666666 245.0,118.66666666666666 245.0,152.0"  fill="none" stroke="currentColor"/>
        <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
            <textPath xlink:href="#arrow-9f9bcae06920463e98e7992840fd8b54-0-0" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">csubj</textPath>
        </text>
        <path class="displacy-arrowhead" d="M50,154.0 L46,146.0 54,146.0"  fill="currentColor"/>
    </g>

    <g class="displacy-arrow">
        <path class="displacy-arc" id="arrow-9f9bcae06920463e98e7992840fd8b54-0-1" stroke-width="2px" d="M70,152.0 70,135.33333333333334 150,135.33333333333334 150,152.0"  fill="none" stroke="currentColor"/>
        <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
            <textPath xlink:href="#arrow-9f9bcae06920463e98e7992840fd8b54-0-1" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">obj</textPath>
        </text>
        <path class="displacy-arrowhead" d="M150,154.0 L146,146.0 154,146.0"  fill="currentColor"/>
    </g>

    <g class="displacy-arrow">
        <path class="displacy-arc" id="arrow-9f9bcae06920463e98e7992840fd8b54-0-2" stroke-width="2px" d="M350,152.0 350,135.33333333333334 440.0,135.33333333333334 440.0,152.0"  fill="none" stroke="currentColor"/>
        <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
            <textPath xlink:href="#arrow-9f9bcae06920463e98e7992840fd8b54-0-2" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">compound</textPath>
        </text>
        <path class="displacy-arrowhead" d="M350,154.0 L346,146.0 354,146.0"  fill="currentColor"/>
    </g>

    <g class="displacy-arrow">
        <path class="displacy-arc" id="arrow-9f9bcae06920463e98e7992840fd8b54-0-3" stroke-width="2px" d="M270,152.0 270,118.66666666666666 450,118.66666666666666 450,152.0"  fill="none" stroke="currentColor"/>
        <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
            <textPath xlink:href="#arrow-9f9bcae06920463e98e7992840fd8b54-0-3" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">obj</textPath>
        </text>
        <path class="displacy-arrowhead" d="M450,154.0 L446,146.0 454,146.0"  fill="currentColor"/>
    </g>

    <g class="displacy-arrow">
        <path class="displacy-arc" id="arrow-9f9bcae06920463e98e7992840fd8b54-0-4" stroke-width="2px" d="M550,152.0 550,118.66666666666666 745.0,118.66666666666666 745.0,152.0"  fill="none" stroke="currentColor"/>
        <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
            <textPath xlink:href="#arrow-9f9bcae06920463e98e7992840fd8b54-0-4" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">case</textPath>
        </text>
        <path class="displacy-arrowhead" d="M550,154.0 L546,146.0 554,146.0"  fill="currentColor"/>
    </g>

    <g class="displacy-arrow">
        <path class="displacy-arc" id="arrow-9f9bcae06920463e98e7992840fd8b54-0-5" stroke-width="2px" d="M650,152.0 650,135.33333333333334 740.0,135.33333333333334 740.0,152.0"  fill="none" stroke="currentColor"/>
        <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
            <textPath xlink:href="#arrow-9f9bcae06920463e98e7992840fd8b54-0-5" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">det</textPath>
        </text>
        <path class="displacy-arrowhead" d="M650,154.0 L646,146.0 654,146.0"  fill="currentColor"/>
    </g>

    <g class="displacy-arrow">
        <path class="displacy-arc" id="arrow-9f9bcae06920463e98e7992840fd8b54-0-6" stroke-width="2px" d="M270,152.0 270,102.0 750,102.0 750,152.0"  fill="none" stroke="currentColor"/>
        <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
            <textPath xlink:href="#arrow-9f9bcae06920463e98e7992840fd8b54-0-6" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">obl</textPath>
        </text>
        <path class="displacy-arrowhead" d="M750,154.0 L746,146.0 754,146.0"  fill="currentColor"/>
    </g>
    </svg>
</div>
</body>
</html>
'''
SECOND_SAMPLE_SVG = '''
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:lang="en" id="c5c065bab8a9425abeb6c747a5231a07-0" class="displacy" width="750" height="402.0" direction="ltr" style="max-width: none; height: 402.0px; color: #000000; background: #ffffff; font-family: Arial; direction: ltr; display: block; overflow: visible">
<style> .bolded{font-weight: bold;} </style>
<text class="displacy-token" fill="currentColor" text-anchor="middle" y="132.0">
    <tspan class="bolded" fill="#66CCEE" x="50">Banning</tspan>

   <tspan class="displacy-tag" dy="2em" fill="currentColor" x="50">VERB</tspan>
  <tspan class="displacy-word" dy="2em" fill="#66CCEE" x=50>Act.</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="132.0">
    <tspan class="bolded" fill="#66CCEE" x="150">tennis</tspan>

   <tspan class="displacy-tag" dy="2em" fill="currentColor" x="150">NOUN</tspan>
  <tspan class="displacy-word" dy="2em" fill="#66CCEE" x=150>Thi.</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="132.0">
    <tspan class="displacy-word" fill="currentColor" x="250">resulted</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="250">VERB</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="132.0">
    <tspan class="displacy-word" fill="currentColor" x="350">in</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="350">ADP</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="132.0">
    <tspan class="displacy-word" fill="currentColor" x="450">players</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="450">NOUN</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="132.0">
    <tspan class="bolded" fill="#4477AA" x="550">banning</tspan>

   <tspan class="displacy-tag" dy="2em" fill="currentColor" x="550">VERB</tspan>
  <tspan class="displacy-word" dy="2em" fill="#4477AA" x=550>Act.</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="132.0">
    <tspan class="bolded" fill="#4477AA" x="650">people .</tspan>

   <tspan class="displacy-tag" dy="2em" fill="currentColor" x="650">NOUN</tspan>
  <tspan class="displacy-word" dy="2em" fill="#4477AA" x=650>Thi.</tspan>
</text>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-c5c065bab8a9425abeb6c747a5231a07-0-0" stroke-width="2px" d="M50,102.0 50,68.66666666666666 250.0,68.66666666666666 250.0,102.0"  fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-c5c065bab8a9425abeb6c747a5231a07-0-0" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">csubj</textPath>
    </text>
    <path class="displacy-arrowhead" d="M50,104.0 L46,96.0 54,96.0"  fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-c5c065bab8a9425abeb6c747a5231a07-0-1" stroke-width="2px" d="M70,102.0 70,85.33333333333333 150,85.33333333333333 150,102.0"  fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-c5c065bab8a9425abeb6c747a5231a07-0-1" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">obj</textPath>
    </text>
    <path class="displacy-arrowhead" d="M150,104.0 L146,96.0 154,96.0"  fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-c5c065bab8a9425abeb6c747a5231a07-0-2" stroke-width="2px" d="M350,102.0 350,85.33333333333333 445.0,85.33333333333333 445.0,102.0"  fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-c5c065bab8a9425abeb6c747a5231a07-0-2" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">case</textPath>
    </text>
    <path class="displacy-arrowhead" d="M350,104.0 L346,96.0 354,96.0"  fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-c5c065bab8a9425abeb6c747a5231a07-0-3" stroke-width="2px" d="M270,102.0 270,68.66666666666666 450,68.66666666666666 450,102.0"  fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-c5c065bab8a9425abeb6c747a5231a07-0-3" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">obl</textPath>
    </text>
    <path class="displacy-arrowhead" d="M450,104.0 L446,96.0 454,96.0"  fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-c5c065bab8a9425abeb6c747a5231a07-0-4" stroke-width="2px" d="M470,102.0 470,85.33333333333333 550,85.33333333333333 550,102.0"  fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-c5c065bab8a9425abeb6c747a5231a07-0-4" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">acl</textPath>
    </text>
    <path class="displacy-arrowhead" d="M550,104.0 L546,96.0 554,96.0"  fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-c5c065bab8a9425abeb6c747a5231a07-0-5" stroke-width="2px" d="M570,102.0 570,85.33333333333333 650,85.33333333333333 650,102.0"  fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-c5c065bab8a9425abeb6c747a5231a07-0-5" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">obj</textPath>
    </text>
    <path class="displacy-arrowhead" d="M650,104.0 L646,96.0 654,96.0"  fill="currentColor"/>
</g>
</svg>
'''
# FIXED_HTML = html.escape(SAMPLE_HTML)
html_string = "<h3>Enter a text below, along with your Semgrex query of choice.</h3>"
st.markdown(html_string, unsafe_allow_html=True)
input_txt = st.text_area('Text to analyze', '''''', placeholder="Banning opal removed artifact decks from the meta.")
input_queries = st.text_area('Semgrex search queries (separate each query with a comma)', placeholder='''{pos:NN}=object <obl {}=action, {cpos:NOUN}=thing <obj {cpos:VERB}=action''')

clicked = st.button("Load Semgrex search visualization")  # use the on_click param

if clicked:
    # components.html(SAMPLE_HTML, height=400, width=1000, scrolling=True)
    # components.html(SECOND_SAMPLE_SVG, height=400, width=1000, scrolling=True)
    if not input_txt:
        st.error("Please provide a text input.")
    elif not input_queries:
        st.error("Please provide a set of Semgrex queries.")
    else:   # no input errors
        try:
            with st.spinner('Processing...'):
                queries = [query.strip() for query in input_queries.split(",")]
                html_strings = visualize_search_str(input_txt, queries, 'en')

                if len(html_strings) == 0:
                    st.write("No Semgrex match hits!")

                for s in html_strings:
                    s_no_overflow = edit_html_overflow(s)
                    components.html(s_no_overflow, height=300, width=1000, scrolling=True)

                if len(html_strings) == 1:
                    st.success(f'Completed! Visualized {len(html_strings)} Semgrex search hit.')
                else:
                    st.success(f'Completed! Visualized {len(html_strings)} Semgrex search hits.')
        except OSError:
            st.error("Your text input or your provided Semgrex queries are incorrect. Please try again.")



