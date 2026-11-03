/**
 * crear_formulario_individual.gs
 * Encuesta individual de Assignum: cada estudiante la responde solo, sin que su grupo participe.
 * Dirigida a estudiantes de carreras de programación de varias universidades, desde 1.er ciclo.
 *
 * Flujo: consentimiento → filtro (18+ y al menos un proyecto grupal reciente) → contexto →
 * perfil (Parte A) → sus últimas 3 tareas en proyectos grupales, de la más reciente a la más antigua.
 * Cada tarea registrada es un par (estudiante, tarea) del dataset; la fila de la hoja es el estudiante.
 * El diseño completo está en el documento "Cuestionario Assignum – Recolección de datos reales".
 *
 * Uso:
 *   1. Entra a https://script.google.com → "Nuevo proyecto" (uno aparte del formulario por grupo).
 *   2. Borra el contenido de Código.gs y pega este archivo completo.
 *   3. Completa los datos de CONSENTIMIENTO (título, nombre, universidad, etc.).
 *   4. Selecciona la función crearFormularioIndividual y pulsa "Ejecutar". Acepta los permisos.
 *   5. En "Registro de ejecución" aparecen los enlaces para editar, responder y ver la hoja.
 */

// ── Configuración ────────────────────────────────────────────────────────────

const NUM_TAREAS = 3;

const ROLES = ['Backend', 'Frontend', 'Testing', 'Database', 'Documentation', 'Management'];
const NIVELES = ['Low', 'Medium', 'High'];
const ESCALA = ['1', '2', '3', '4', '5'];

// Rangos de horas estimadas; al convertir a CSV se usan los valores 5, 14, 30 y 45
const HORAS_ESTIMADAS = ['Menos de 8 h', '8 a 20 h', '21 a 40 h', 'Más de 40 h'];

const CARRERAS = [
  'Ingeniería de Software',
  'Ingeniería de Sistemas',
  'Ciencias de la Computación',
  'Ingeniería Informática',
];

const CONSENTIMIENTO = [
  'Estás invitado(a) a participar en la investigación "[título de la tesis]", realizada por ' +
    '[tu nombre], estudiante de Ingeniería de Software de [universidad], bajo la asesoría de [asesor].',
  '',
  'Propósito: estudiar qué factores influyen en el buen desempeño de tareas en proyectos grupales ' +
    'de software, para construir un sistema que recomiende asignaciones de tareas.',
  '',
  'Qué harás: responder preguntas sobre tu perfil y sobre tus últimas 3 tareas en proyectos ' +
    'grupales (unos 15 minutos). No necesitas que tu grupo participe.',
  '',
  'Confidencialidad: la encuesta es anónima, no se registra tu nombre ni tu correo. Los datos se ' +
    'usarán solo con fines académicos y se publicarán de forma agregada.',
  '',
  'Voluntariedad: tu participación es voluntaria y no afecta ninguna nota. Puedes dejar de ' +
    'responder en cualquier momento.',
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

const ORDINAL = ['', 'más reciente', 'anterior a esa', 'anterior a esa'];

// ── Utilidades ───────────────────────────────────────────────────────────────

function validarNumero(min, max) {
  return FormApp.createTextValidation()
    .setHelpText('Escribe un número entre ' + min + ' y ' + max + '.')
    .requireNumberBetween(min, max)
    .build();
}

function preguntaOpciones(form, titulo, opciones, ayuda) {
  const item = form.addMultipleChoiceItem().setTitle(titulo).setChoiceValues(opciones).setRequired(true);
  if (ayuda) item.setHelpText(ayuda);
  return item;
}

function preguntaNumero(form, titulo, min, max, ayuda) {
  const item = form.addTextItem().setTitle(titulo).setRequired(true).setValidation(validarNumero(min, max));
  if (ayuda) item.setHelpText(ayuda);
  return item;
}

function preguntaEscala(form, titulo, izquierda, derecha, ayuda) {
  const item = form.addScaleItem().setTitle(titulo).setBounds(1, 5).setLabels(izquierda, derecha).setRequired(true);
  if (ayuda) item.setHelpText(ayuda);
  return item;
}

/** Pregunta Sí/No en su propia sección: "Sí" continúa y "No" envía el formulario. */
function filtro(form, tituloSeccion, pregunta) {
  form.addPageBreakItem().setTitle(tituloSeccion);
  const item = form.addMultipleChoiceItem().setTitle(pregunta).setRequired(true);
  item.setChoices([
    item.createChoice('Sí', FormApp.PageNavigationType.CONTINUE),
    item.createChoice('No', FormApp.PageNavigationType.SUBMIT),
  ]);
}

/** Preguntas T1–T13 de una tarea. Los títulos llevan el número para que cada una tenga su columna. */
function preguntasTarea(form, k) {
  const t = 'Tarea ' + k + ' · ';

  // Contexto
  form.addTextItem().setTitle(t + 'T1. Descríbela en una línea').setHelpText('Ej.: "hacer el login con Firebase".');
  form.addTextItem().setTitle(t + 'T2. ¿En qué curso o proyecto fue?');
  preguntaOpciones(form, t + 'T3. ¿Cuándo fue?', ['Este ciclo', 'Ciclo anterior']);
  preguntaNumero(form, t + 'T4. ¿Cuántas personas había en el equipo, contándote?', 2, 15);

  // Características de la tarea
  preguntaOpciones(form, t + 'T5. Tipo de tarea', ROLES, 'Si mezclaba varios, elige el principal.');
  preguntaOpciones(
    form,
    t + 'T6. Complejidad',
    NIVELES,
    'Low = pocas horas, sin dependencias. Medium = varias partes. High = muchas partes, dependencias o algo nuevo para ti.'
  );
  preguntaOpciones(form, t + 'T7. Prioridad para el proyecto', NIVELES);
  preguntaOpciones(form, t + 'T8. ¿Cuánto se esperaba que tomara?', HORAS_ESTIMADAS, 'Si nadie lo estimó, lo que pensabas antes de empezar.');
  preguntaNumero(form, t + 'T9. En esas semanas, ¿cuántas horas por semana podías dedicar al proyecto?', 1, 60);

  // Resultado (define success)
  preguntaOpciones(form, t + 'T10. ¿La entregaste?', ['Sí', 'No']);
  preguntaOpciones(form, t + 'T11. ¿La entregaste dentro del plazo?', ['Sí', 'No']);
  preguntaOpciones(form, t + 'T12. ¿Cómo calificaron esa entrega el docente o el líder?', [
    '1 – Desaprobada o hubo que rehacerla',
    '2 – Aprobada con observaciones importantes',
    '3 – Cumplió lo pedido',
    '4 – Buena, con observaciones menores',
    '5 – Sobresaliente',
  ]);
  preguntaOpciones(
    form,
    t + 'T13. ¿Cómo valoraron tus compañeros tu aporte en esa tarea?',
    ESCALA.concat(['No lo sé']),
    'Por coevaluación, comentarios o cómo reaccionaron. 1 = muy mal, 5 = muy bien.'
  );
}

// ── Formulario ───────────────────────────────────────────────────────────────

function crearFormularioIndividual() {
  const form = FormApp.create('Assignum – Encuesta individual');
  form
    .setDescription(CONSENTIMIENTO)
    .setCollectEmail(false)
    .setAllowResponseEdits(false)
    .setLimitOneResponsePerUser(false)
    .setShowLinkToRespondAgain(false)
    .setConfirmationMessage(
      '¡Gracias por participar! Si conoces a otros estudiantes de carreras de programación, compárteles el enlace.'
    );

  // 1. Consentimiento
  const consentimiento = form
    .addMultipleChoiceItem()
    .setTitle('¿Aceptas participar y autorizas el uso anónimo de tus respuestas?')
    .setRequired(true);
  consentimiento.setChoices([
    consentimiento.createChoice('Sí, acepto', FormApp.PageNavigationType.CONTINUE),
    consentimiento.createChoice('No acepto', FormApp.PageNavigationType.SUBMIT),
  ]);

  // 2. Filtro (cada pregunta en su sección: Forms solo ramifica con una pregunta por sección)
  filtro(form, 'Antes de empezar (1 de 2)', '¿Tienes 18 años o más?');
  filtro(
    form,
    'Antes de empezar (2 de 2)',
    '¿Has hecho al menos un proyecto grupal de software en este ciclo o en el anterior?'
  );

  // 3. Contexto
  form.addPageBreakItem().setTitle('Sobre tus estudios');
  form.addTextItem().setTitle('País').setRequired(true);
  form.addTextItem().setTitle('Universidad').setRequired(true).setHelpText('Escribe el nombre completo, sin siglas.');
  preguntaOpciones(form, 'Tipo de universidad', ['Pública', 'Privada']);
  form.addMultipleChoiceItem().setTitle('Carrera').setChoiceValues(CARRERAS).showOtherOption(true).setRequired(true);
  preguntaNumero(form, 'Ciclo o semestre actual', 1, 14);
  preguntaOpciones(form, '¿Trabajas además de estudiar?', ['No', 'Medio tiempo', 'Tiempo completo']);

  // 4. Perfil (Parte A)
  form
    .addPageBreakItem()
    .setTitle('Tu perfil · Competencias técnicas')
    .setHelpText('¿Qué nivel tienes HOY en cada área?\n\n' + AYUDA_TECNICAS);
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
    .setTitle('Tu perfil · Competencias colaborativas')
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

  form.addPageBreakItem().setTitle('Tu perfil · Experiencia');
  preguntaNumero(
    form,
    'A12. ¿Cuántos proyectos de software grupales has completado?',
    0,
    30,
    'Cuenta cursos, prácticas, hackatones y trabajo.'
  );
  preguntaOpciones(form, 'A14. ¿Qué rol sueles tener en los proyectos grupales?', ROLES);
  preguntaEscala(form, 'A15. ¿Qué tan bien te suele ir en ese rol?', 'Muy mal', 'Muy bien');
  preguntaEscala(
    form,
    'B1. En general, ¿cómo suelen evaluarte tus compañeros en los trabajos grupales?',
    'Muy mal',
    'Muy bien',
    'Piensa en coevaluaciones o en lo que te han dicho.'
  );

  // 5. Tareas: las últimas NUM_TAREAS, de la más reciente a la más antigua
  let otra = null;
  for (let k = 1; k <= NUM_TAREAS; k++) {
    const pagina = form
      .addPageBreakItem()
      .setTitle('Tarea ' + k + ' de ' + NUM_TAREAS + ' · la ' + ORDINAL[k])
      .setHelpText(
        'Piensa en la tarea ' + ORDINAL[k] + ' que te asignaron A TI en un proyecto grupal de software ' +
          'de este ciclo o el anterior. Cuéntala aunque haya salido mal: es igual de importante.'
      );
    if (otra) {
      otra.setChoices([
        otra.createChoice('Sí', pagina),
        otra.createChoice('No, terminar y enviar', FormApp.PageNavigationType.SUBMIT),
      ]);
    }
    preguntasTarea(form, k);
    otra = k < NUM_TAREAS
      ? form
          .addMultipleChoiceItem()
          .setTitle('¿Tuviste otra tarea anterior a esta, en este ciclo o el anterior?')
          .setRequired(true)
      : null;
  }

  // Hoja de cálculo para las respuestas
  const hoja = SpreadsheetApp.create('Assignum – Respuestas individuales');
  form.setDestination(FormApp.DestinationType.SPREADSHEET, hoja.getId());

  Logger.log('Editar formulario: ' + form.getEditUrl());
  Logger.log('Enlace para responder: ' + form.getPublishedUrl());
  Logger.log('Hoja de respuestas: ' + hoja.getUrl());
}
