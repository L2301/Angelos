const siteHeader = document.querySelector('.site-header');
const menuToggle = document.querySelector('.menu-toggle');
const siteNavigation = document.querySelector('#site-navigation');
const mobileViewport = window.matchMedia('(max-width: 760px)');

function setMenuOpen(isOpen, returnFocus = false) {
  siteHeader.classList.toggle('menu-open', isOpen);
  menuToggle.setAttribute('aria-expanded', String(isOpen));
  menuToggle.firstChild.textContent = isOpen ? 'Close ' : 'Menu ';
  menuToggle.querySelector('span').textContent = isOpen ? '−' : '+';
  if (returnFocus) menuToggle.focus();
}

menuToggle.hidden = false;
siteHeader.classList.add('menu-ready');
menuToggle.addEventListener('click', () => {
  setMenuOpen(menuToggle.getAttribute('aria-expanded') !== 'true');
});
siteHeader.addEventListener('keydown', event => {
  if (event.key === 'Escape' && menuToggle.getAttribute('aria-expanded') === 'true') {
    setMenuOpen(false, true);
  }
});
siteNavigation.addEventListener('click', event => {
  if (mobileViewport.matches && event.target.closest('a')) setMenuOpen(false, true);
});
mobileViewport.addEventListener('change', () => {
  const navigationHasFocus = siteNavigation.contains(document.activeElement);
  const toggleHasFocus = document.activeElement === menuToggle;
  setMenuOpen(false, mobileViewport.matches && navigationHasFocus);
  if (!mobileViewport.matches && toggleHasFocus) siteNavigation.querySelector('a').focus();
});

const stepControls = document.querySelector('.step-controls');
const stepButtons = [...document.querySelectorAll('[data-step]')];
const stagePanels = [...document.querySelectorAll('.stage-panel')];
const diagramNodes = [...document.querySelectorAll('[data-diagram-stage]')];
const stepAnnouncement = document.querySelector('#step-announcement');

function selectStep(selectedIndex, announce = true) {
  stepButtons.forEach((button, buttonIndex) => {
    button.setAttribute('aria-pressed', String(buttonIndex === selectedIndex));
  });
  stagePanels.forEach((panel, panelIndex) => {
    panel.hidden = panelIndex !== selectedIndex;
  });
  diagramNodes.forEach((node, nodeIndex) => {
    node.classList.toggle('is-current', nodeIndex === selectedIndex);
  });
  if (announce) {
    const selectedPanel = stagePanels[selectedIndex];
    stepAnnouncement.textContent = `Step ${selectedIndex + 1} of 5. ${selectedPanel.querySelector('h3').textContent} ${selectedPanel.querySelector('.status-label').textContent}. ${selectedPanel.querySelector('p').textContent} ${selectedPanel.querySelector('.detail-boundary').textContent}`;
  }
}

stepButtons.forEach(button => {
  button.addEventListener('click', () => selectStep(Number(button.dataset.step)));
});
selectStep(0, false);
stepControls.hidden = false;
document.querySelector('.stage-details').classList.add('is-interactive');

const campaignConcepts = {
  quiet: {
    label: 'The quiet explorer',
    eyebrow: 'Less noise. More outside.',
    headline: ['Find your', 'own quiet.'],
    copy: 'A little further from the everyday.',
    variable: 'Quiet pace · spacious composition · reflective copy',
    number: '01',
  },
  bold: {
    label: 'The restless explorer',
    eyebrow: 'Take the long way home.',
    headline: ['Go where', 'you feel it.'],
    copy: 'A new perspective is just over the ridge.',
    variable: 'Restless pace · sharper contrast · adventurous copy',
    number: '02',
  },
};

const personaButtons = [...document.querySelectorAll('[data-persona]')].filter(element => element.tagName === 'BUTTON');
const campaignPreview = document.querySelector('#campaign-preview');
const campaignHeadline = document.querySelector('#campaign-headline');
const conceptAnnouncement = document.querySelector('#concept-announcement');

function selectPersona(persona) {
  const concept = campaignConcepts[persona];
  if (!concept) return;
  campaignPreview.dataset.persona = persona;
  personaButtons.forEach(button => {
    button.setAttribute('aria-pressed', String(button.dataset.persona === persona));
  });
  campaignHeadline.replaceChildren(document.createTextNode(concept.headline[0]), document.createElement('br'), document.createTextNode(concept.headline[1]));
  document.querySelector('#campaign-eyebrow').textContent = concept.eyebrow;
  document.querySelector('#campaign-copy').textContent = concept.copy;
  document.querySelector('#brief-variable').textContent = concept.variable;
  document.querySelector('#concept-number').textContent = concept.number;
  conceptAnnouncement.textContent = `${concept.label}. ${concept.headline.join(' ')} ${concept.copy} ${concept.variable}. Illustrative concept, not model output.`;
}

personaButtons.forEach(button => {
  button.addEventListener('click', () => selectPersona(button.dataset.persona));
});
document.querySelector('.persona-controls').hidden = false;
