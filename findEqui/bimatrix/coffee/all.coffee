$(document).ready ->

  setAttributes = (el, attrs) ->
    for key, value of attrs
      el.setAttribute(key, value)

  create_matrix = (rows, cols) ->
    toClone = document.createElement('td')
    toClone.className = 'cell'
    bimatrix = document.getElementById('bimatrix-table')
    bimatrix.innerHTML = ''
    i = 0
    while i < rows
      row = document.createElement('tr')
      j = 0
      while j < cols
        cell = toClone.cloneNode()
        inp1 = document.createElement('input')
        setAttributes inp1,
          'row': i
          'col': j
          'required': 'true'
          'type': 'number'
          'min': '0'
        inp2 = inp1.cloneNode()
        inp1.className = 'A_entry'
        inp2.className = 'B_entry'
        cell.appendChild(inp1).className = 'A_entry'
        cell.appendChild(inp2).className = 'B_entry'
        row.appendChild cell
        j++
      bimatrix.appendChild row
      i++
    $('form#bimatrix-form input[name="hidden_m"]').val rows
    $('form#bimatrix-form input[name="hidden_n"]').val cols
    return

  create_matrix 2, 2

  $('form.dimensions').on 'submit', ->
    m = $('input#number_m').val()
    n = $('input#number_n').val()
    create_matrix m, n

  $('form.dimensions input').on 'focus', ->
    $(this).val('')

  $('form#bimatrix-form .random').on 'click', ->
    $.each $('form#bimatrix-form').find(':input:not([type=hidden])'), (index, input) ->
      rand = Math.floor(Math.random() * 10) + 1
      $(input).val rand

  $('form#bimatrix-form .clear').on 'click', ->
    $('form#bimatrix-form').find(':input:not([type=hidden])').val('')

  $('form#bimatrix-form').on 'submit', ->
    $('.results').hide()

    build_equilbria_table = (equilibria) ->
      eq_table = $('#eq-table tbody')[0]
      eq_table.innerHTML = ''

      for eq, i in equilibria
        row = create_element 'tr'
        eq_table.appendChild row
        create_element('td', i+1, null, row)
        create_element('td', 'x<sup>' + eq[0]['number'] + '</sup>', null, row)
        create_element('td','[' + eq[0]['distribution'].join(', ') + ']', null, row)
        create_element('td', eq[0]['payoff'], null, row)
        create_element('td', 'y<sup>' + eq[1]['number'] + '</sup>', null, row)
        create_element('td', '[' + eq[1]['distribution'].join(', ') + ']', null, row)
        create_element('td', eq[1]['payoff'], null, row)

    create_element = (type, html = null, class_name = null, parent = null)->
      result = document.createElement(type)
      result.innerHTML = html if html != null
      result.className = class_name if class_name
      parent.appendChild result if parent
      return result

    build_components_table = (results) ->
      comp_table = $('#comp-table tbody')[0]
      comp_table.innerHTML = ''

      for comp_value, i in results
        row = create_element 'tr', null, null, comp_table
        create_element 'td', i+1, null, row
        subsets_cell = create_element 'td', null, 'subsets-cell', row
        subsets_cell.setAttribute('colspan', 3)
        equilibria_cell = create_element 'td', null, 'equilibria-cell', row
        equilibria_cell.setAttribute('colspan', 2)
        create_element 'td', comp_value['index'], null, row

        subsets = comp_value['nash_subsets']
        equilibria = comp_value['equilibria']

        subsets_table = create_element 'table', null, 'small-table subsets', subsets_cell
        subsets_tbody = create_element 'tbody', null, null, subsets_table

        for subset in subsets
          current_row = document.createElement('tr')
          current_row.appendChild parse_component('x', subset[0])
          cell = create_element 'td', '&times', 'central', current_row
          current_row.appendChild parse_component('y', subset[1])
          subsets_tbody.appendChild current_row

        equilibria_table = create_element 'table', null, 'small-table', equilibria_cell
        equilibria_tbody = create_element 'tbody', null, null, equilibria_table

        for eq in equilibria
          current_row = create_element 'tr', null, null, equilibria_tbody
          create_element 'td', eq['eq_number'], 'x', current_row
          create_element 'td', eq['lex_index'], 'y', current_row

    parse_component = (player, strategies) ->
      text = '{ '
      for i in strategies
        text += player + '<sup>' + i + '</sup>' + ", "

      text = text.slice(0, -2) + ' }'
      result = document.createElement('td')
      result.className = player
      result.innerHTML = text
      return result

    collect_matrix_data = (rows, cols) ->
      A_values = []
      B_values = []
      form = $('form#bimatrix-form')
      i = 0
      while i < rows
        A_values.push []
        B_values.push []
        j = 0
        while j < cols
          A_values[i].push form.find('input[row=' + i + '][col=' + j + '].A_entry').val()
          B_values[i].push form.find('input[row=' + i + '][col=' + j + '].B_entry').val()
          j++
        i++
      [A_values, B_values]

    rows = $(this).find('input[name="hidden_m"]').val()
    cols = $(this).find('input[name="hidden_n"]').val()
    matrices = collect_matrix_data(parseInt(rows), parseInt(cols))

    $.ajax
      type: 'POST'
      url: '/'
      data:
        'A': JSON.stringify(matrices[0])
        'B': JSON.stringify(matrices[1])
        'm': rows
        'n': cols
      success: (results) ->
        build_equilbria_table results['equilibria']
        build_components_table results['components']
        $('.results').fadeIn()
      error: (error) ->
        console.log error

  timer = null
  $(document).ajaxStart ->
    if timer then clearTimeout(timer)
    timer = setTimeout((-> $('body').addClass 'loading'), 500)
  $(document).ajaxComplete ->
    clearTimeout(timer)
    $('body').removeClass 'loading'
