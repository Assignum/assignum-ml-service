/**
 * crear_formulario.gs
 * Crea en Google Forms el formulario único de recolección de datos de Assignum
 * y una hoja de cálculo vinculada donde llegan todas las respuestas.
 *
 * Uso:
 *   1. Entra a https://script.google.com → "Nuevo proyecto".
 *   2. Borra el contenido de Código.gs y pega este archivo completo.
 *   3. Completa los datos de CONSENTIMIENTO (título, nombre, universidad, etc.).
 *   4. Selecciona la función crearFormulario y pulsa "Ejecutar". Acepta los permisos.
 *   5. En "Registro de ejecución" aparecen los enlaces para editar, responder y ver la hoja.
 *
 * Estructura: una primera sección pide el código del participante y un menú
 * "¿Qué vas a registrar?". Según la opción, el formulario salta a:
 *   - Perfil inicial      → consentimiento, Parte A y evaluación inicial de compañeros (B1)
 *   - Evaluación final    → evaluación final de compañeros (B2)
 *   - Tarea nueva         → Parte C al asignar (líder)
 *   - Cierre de tarea     → Parte C al cerrar (líder)
 * Cada opción termina en "Enviar". Para registrar otra cosa se usa "Enviar otra respuesta".
 */

// ── Configuración ────────────────────────────────────────────────────────────

const MAX_COMPANEROS = 6; // compañeros que se pueden evaluar por envío (equipos de hasta 7)

const CODIGO_ESTUDIANTE = '^G\\d{2}-E\\d{2}$'; // ej. G03-E02
const CODIGO_TAREA = '^G\\d{2}-T\\d{2,3}$'; // ej. G03-T07

const ROLES = ['Backend', 'Frontend', 'Testing', 'Database', 'Documentation', 'Management'];
const NIVELES = ['Low', 'Medium', 'High'];
const ESCALA = ['1', '2', '3', '4', '5'];

const CONSENTIMIENTO = [
  'Estás invitado(a) a participar en la investigación "[título de la tesis]", realizada por ' +
    '[tu nombre], estudiante de Ingeniería de Software de [universidad], bajo la asesoría de [asesor].',
  '',
  'Propósito: estudiar qué factores influyen en el buen desempeño de tareas en proyectos grupales ' +
    'de software, para construir un sistema que recomiende asignaciones de tareas.',
  '',
  'Qué harás: responder un cuestionario de perfil (unos 10 minutos), evaluar a tus compañeros de ' +
    'equipo y registrar brevemente las tareas que completes durante el sprint.',
  '',
  'Confidencialidad: tus respuestas se identifican con un código, no con tu nombre. Los datos se ' +
    'usarán solo con fines académicos y se publicarán de forma agregada. Tus evaluaciones de pares ' +
    'no se compartirán con tus compañeros.',
  '',
  'Voluntariedad: tu participación es voluntaria y no afecta tu nota del curso. Puedes retirarte ' +
    'en cualquier momento sin dar explicaciones.',
  '',
  'Contacto: [correo del investigador].',
].join('\n');

const AYUDA_TECNICAS = [
  '1 = Nunca lo he hecho o no lo conozco.',
  '2 = Lo he hecho una vez con mucha ayuda.',
  '3 = Puedo hacerlo solo en tareas sencillas, consultando documentación.',
  '4 = Lo hago con soltura en tareas medianas y ayudo a otros.',
  '5 = Resuelvo tareas complejas y podría enseñarlo.',
].join('\n');

// ── Utilidades ───────────────────────────────────────────────────────────────

function validarPatron(patron, ejemplo) {
  return FormApp.createTextValidation()
    .setHelpText('Formato esperado: ' + ejemplo)
    .requireTextMatchesPattern(patron)
    .build();
}

function validarNumero(min, max) {
  return FormApp.createTextValidation()
    .setHelpText('Escribe un número entre ' + min + ' y ' + max + '.')
    .requireNumberBetween(min, max)
    .build();
}

function preguntaCodigo(form, titulo, patron, ejemplo) {
  return form
    .addTextItem()
    .setTitle(titulo)
    .setRequired(true)
    .setValidation(validarPatron(patron, ejemplo));
}

function preguntaOpciones(form, titulo, opciones, ayuda) {
  const item = form.addMultipleChoiceItem().setTitle(titulo).setChoiceValues(opciones).setRequired(true);
  if (ayuda) item.setHelpText(ayuda);
  return item;
}

/**
 * Crea MAX_COMPANEROS secciones seguidas para evaluar compañeros. Al final de cada una se
 * pregunta "¿Quieres evaluar a otro compañero?": Sí → siguiente sección, No → enviar.
 * Los títulos llevan el número del compañero para que cada uno tenga su columna en la hoja.
 * Devuelve el salto de página de la primera sección.
 */
function bloqueCompaneros(form, titulo, ayuda, agregarPreguntas) {
  let primera = null;
  let otra = null;
  for (let k = 1; k <= MAX_COMPANEROS; k++) {
    const pagina = form
      .addPageBreakItem()
      .setTitle(titulo + ' (compañero ' + k + ')')
      .setHelpText(ayuda);
    if (!primera) primera = pagina;
    if (otra) {
      otra.setChoices([
        otra.createChoice('Sí', pagina),
        otra.createChoice('No, terminar y enviar', FormApp.PageNavigationType.SUBMIT),
      ]);
    }
    preguntaCodigo(form, 'Compañero ' + k + ' · código', CODIGO_ESTUDIANTE, 'G03-E02');
    agregarPreguntas(form, k);
    otra = k < MAX_COMPANEROS
      ? form.addMultipleChoiceItem().setTitle('¿Quieres evaluar a otro compañero?').setRequired(true)
      : null;
  }
  return primera;
}

// ── Formulario ───────────────────────────────────────────────────────────────

function crearFormulario() {
  const form = FormApp.create('Assignum – Recolección de datos');
  form
    .setDescription(
      'Formulario de la investigación sobre asignación de tareas en proyectos de software. ' +
        'Úsalo cada vez que tengas algo que registrar: elige la opción en el menú de abajo.'
    )
    .setCollectEmail(false)
    .setAllowResponseEdits(false)
    .setLimitOneResponsePerUser(false)
    .setShowLinkToRespondAgain(true)
    .setConfirmationMessage('¡Gracias! Si tienes algo más que registrar, pulsa "Enviar otra respuesta".');

  // Sección 1: identificación y menú
  preguntaCodigo(form, 'Tu código de participante', CODIGO_ESTUDIANTE, 'G03-E02').setHelpText(
    'Te lo da el investigador. G03-E02 = grupo 3, estudiante 2.'
  );
  const menu = form.addMultipleChoiceItem().setTitle('¿Qué vas a registrar?').setRequired(true);

  // ── Perfil inicial ─────────────────────────────────────────────────────────
  const pConsentimiento = form
    .addPageBreakItem()
    .setTitle('Consentimiento informado')
    .setHelpText(CONSENTIMIENTO);
  const consentimiento = form
    .addMultipleChoiceItem()
    .setTitle('¿Aceptas participar y autorizas el uso anónimo de tus respuestas?')
    .setRequired(true);
  consentimiento.setChoices([
    consentimiento.createChoice('Sí, acepto', FormApp.PageNavigationType.CONTINUE),
    consentimiento.createChoice('No acepto', FormApp.PageNavigationType.SUBMIT),
  ]);

  form
    .addPageBreakItem()
    .setTitle('Parte A · Competencias técnicas')
    .setHelpText('¿Qué nivel tienes en cada área?\n\n' + AYUDA_TECNICAS);
  form
    .addGridItem()
    .setTitle('Nivel técnico')
    .setRows([
      'A1. Backend (APIs, lógica de servidor: Node, Django, Spring)',
      'A2. Frontend / móvil (interfaces web o móviles: React, Flutter)',
      'A3. Bases de datos (modelado, SQL, consultas, migraciones)',
      'A4. Testing (pruebas unitarias, de integración, casos de prueba)',
      'A5. Documentación (manuales, diagramas, requisitos)',
      'A6. Git y GitHub (ramas, pull requests, resolver conflictos)',
      'A7. Metodologías ágiles (Scrum, sprints, historias de usuario)',
    ])
    .setColumns(ESCALA)
    .setRequired(true);

  form
    .addPageBreakItem()
    .setTitle('Parte A · Competencias colaborativas')
    .setHelpText('¿Qué tan de acuerdo estás con cada frase? 1 = nada de acuerdo, 5 = totalmente de acuerdo.');
  form
    .addGridItem()
    .setTitle('Trabajo con otros')
    .setRows([
      'A8. Coordino mi trabajo con otros y cumplo lo que acuerdo con el equipo.',
      'A9. Aviso a tiempo cuando tengo un bloqueo o un retraso.',
      'A10. He organizado al equipo, repartido tareas o tomado decisiones por el grupo.',
      'A11. Planifico mis tareas y suelo terminar antes del plazo.',
      'A18. Control de atención: en esta fila marca el 2.',
    ])
    .setColumns(ESCALA)
    .setRequired(true);

  form.addPageBreakItem().setTitle('Parte A · Experiencia y disponibilidad');
  form
    .addTextItem()
    .setTitle('A12. ¿Cuántos proyectos de software grupales has completado?')
    .setHelpText('Cuenta cursos, prácticas, hackatones y trabajo.')
    .setRequired(true)
    .setValidation(validarNumero(0, 30));
  form
    .addTextItem()
    .setTitle('A13. En este sprint, ¿cuántas horas por semana puedes dedicar realmente al proyecto?')
    .setRequired(true)
    .setValidation(validarNumero(1, 60));
  preguntaOpciones(form, 'A14. En tu último proyecto grupal, ¿cuál fue tu rol principal?', ROLES);
  form
    .addScaleItem()
    .setTitle('A15. ¿Qué tan bien crees que te fue en ese rol?')
    .setBounds(1, 5)
    .setLabels('Muy mal', 'Muy bien')
    .setRequired(true);
  form.addTextItem().setTitle('A16. Ciclo o semestre actual').setRequired(true).setValidation(validarNumero(1, 14));
  preguntaOpciones(form, 'A17. ¿Trabajas además de estudiar?', ['No', 'Medio tiempo', 'Tiempo completo']);

  bloqueCompaneros(
    form,
    'Parte B1 · Evaluación inicial',
    'Evalúa a cada compañero de tu equipo actual, uno por sección. Tus respuestas son confidenciales.',
    function (f, k) {
      preguntaOpciones(
        f,
        'Compañero ' + k + ' · B1. En trabajos anteriores, ¿qué tan confiable fue su aporte al equipo?',
        ESCALA.concat(['No he trabajado antes con esta persona']),
        '1 = nada confiable, 5 = totalmente confiable.'
      );
    }
  );

  // ── Evaluación final ───────────────────────────────────────────────────────
  const pFinal = bloqueCompaneros(
    form,
    'Parte B2 · Evaluación final',
    'Evalúa a cada compañero por su trabajo en este sprint. 1 = nada, 5 = totalmente. ' +
      'Tus respuestas son confidenciales.',
    function (f, k) {
      f.addGridItem()
        .setTitle('Compañero ' + k + ' · B2')
        .setRows([
          'B2.1 ¿Cumplió con las tareas que se le asignaron?',
          'B2.2 ¿La calidad de su trabajo fue la esperada?',
          'B2.3 ¿Comunicó avances y bloqueos a tiempo?',
          'B2.4 ¿Volverías a trabajar con esta persona?',
        ])
        .setColumns(ESCALA)
        .setRequired(true);
    }
  );
  pFinal.setGoToPage(FormApp.PageNavigationType.SUBMIT); // el perfil inicial termina antes de B2

  // ── Tarea nueva (líder) ────────────────────────────────────────────────────
  const pAsignar = form
    .addPageBreakItem()
    .setTitle('Parte C · Registrar una tarea nueva')
    .setHelpText('Llénalo al asignar la tarea, antes de que se empiece a trabajar en ella.');
  pAsignar.setGoToPage(FormApp.PageNavigationType.SUBMIT); // la evaluación final termina antes de C
  preguntaCodigo(form, 'C1. Código de la tarea', CODIGO_TAREA, 'G03-T07');
  preguntaCodigo(form, 'C2. Código del estudiante asignado', CODIGO_ESTUDIANTE, 'G03-E02');
  preguntaOpciones(form, 'C3. Tipo de tarea', ROLES);
  preguntaOpciones(
    form,
    'C4. Complejidad',
    NIVELES,
    'Low = pocas horas, sin dependencias. High = varias partes, dependencias o algo nuevo para el equipo.'
  );
  preguntaOpciones(form, 'C5. Prioridad', NIVELES);
  form.addTextItem().setTitle('C6. Horas estimadas').setRequired(true).setValidation(validarNumero(1, 80));
  form.addDateItem().setTitle('C7. Fecha límite').setRequired(true);
  preguntaOpciones(form, 'C8. ¿Cómo se asignó la tarea?', [
    'La eligió el estudiante',
    'La asignó el líder',
    'Sorteo',
    'Recomendación de Assignum',
  ]);

  // ── Cierre de tarea (líder) ────────────────────────────────────────────────
  const pCerrar = form
    .addPageBreakItem()
    .setTitle('Parte C · Cerrar una tarea')
    .setHelpText('Llénalo cuando venza el plazo o se entregue la tarea. C12 y C13 pregúntalos al estudiante.');
  pCerrar.setGoToPage(FormApp.PageNavigationType.SUBMIT); // el registro de tarea nueva termina antes
  preguntaCodigo(form, 'C1. Código de la tarea', CODIGO_TAREA, 'G03-T07');
  preguntaOpciones(form, 'C9. ¿Se entregó?', ['Sí', 'No']);
  preguntaOpciones(form, 'C10. ¿Se entregó dentro del plazo?', ['Sí', 'No']);
  form
    .addScaleItem()
    .setTitle('C11. Calidad del resultado')
    .setHelpText('1 = hubo que rehacerla, 3 = cumple lo pedido, 5 = superó lo esperado.')
    .setBounds(1, 5)
    .setLabels('Hubo que rehacerla', 'Superó lo esperado')
    .setRequired(true);
  form.addTextItem().setTitle('C12. Horas reales dedicadas').setRequired(true).setValidation(validarNumero(0, 200));
  preguntaOpciones(form, 'C13. ¿Recibió ayuda importante de otro compañero?', ['Sí', 'No']);

  // Menú de la primera sección (se arma al final porque necesita las secciones ya creadas)
  menu.setChoices([
    menu.createChoice('Perfil inicial (primera vez, al inicio del sprint)', pConsentimiento),
    menu.createChoice('Evaluación final de mis compañeros (fin del sprint)', pFinal),
    menu.createChoice('Registrar una tarea nueva (líder del equipo)', pAsignar),
    menu.createChoice('Cerrar una tarea terminada (líder del equipo)', pCerrar),
  ]);

  // Hoja de cálculo para las respuestas
  const hoja = SpreadsheetApp.create('Assignum – Respuestas');
  form.setDestination(FormApp.DestinationType.SPREADSHEET, hoja.getId());

  Logger.log('Editar formulario: ' + form.getEditUrl());
  Logger.log('Enlace para responder: ' + form.getPublishedUrl());
  Logger.log('Hoja de respuestas: ' + hoja.getUrl());
}
